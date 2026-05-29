from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from PySide6 import QtCore

from kindred.gui.controllers.simulation_completion_policy import CacheAuthorityState, CompletionPolicyContext
from kindred.gui.controllers.simulation_result_materialization import MaterializedDisplayResult
from kindred.gui.ports import (
    CompletedRunDisplayCoverage,
    CompletedRunDisplayIntent,
    CompletedRunDisplayTransaction,
    CompletionDisplayEntry,
    DisplayStatus,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
    FreshPreviewDisplayEntry,
    SimulationCompletionDisplayOutcome,
)

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
    batch_queue_done: bool = False


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
    warnings: Sequence[Any]
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
    clear_pending_progress_status: Callable[[], None] = lambda: None


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
        debug_batch_parallel: bool,
    ) -> None:
        try:
            logger.info("Simulation completed successfully")
            if bool(debug_batch_parallel):
                logger.info(
                    "BATCH_PAR completion handler run_id=%s request_id=%s set_id=%s slider=%s ts=%.6f",
                    int(state.run_id or 0),
                    int(state.request_id or 0),
                    str(state.batch_set_id or ""),
                    bool(state.slider_triggered),
                    float(perf_counter()),
                )
            completion = self.build_result_state(dict(result))
            self.resolve_cache_key(state)
            self.resolve_result_ownership(completion, state)
            self.apply_materialization_contract(completion, state)
            self.publish_primary_materialization(completion, state)
            self.apply_algebra_status(completion)
            self.publish_cache_truth(state)
            self.publish_cache_entry(completion, state)
            self.record_in_flight_completion_display_entry(completion, state)
            display_outcome = self.publish_display(completion, state)
            if self._non_displayed_outcome_is_terminal(display_outcome, state):
                if isinstance(state.ctx, Mapping):
                    state.ctx = self._batch_context_owner.deactivate_if_active(state.ctx)
                state.batch_queue_done = True
                return
            self._result_materialization_owner.refresh_primary_result_controls(
                mechanism=completion.mechanism,
                energy_mode=bool(completion.energy_mode),
                slider_triggered=bool(state.slider_triggered),
                is_primary=bool(completion.is_primary),
            )
            state.batch_queue_done = self.advance_batch_success(completion, state)
            if state.batch_queue_done:
                self.finalize_success(completion, state, display_outcome=display_outcome)
        except Exception as exc:
            logger.error("Error displaying results: %s", exc, exc_info=True)
            self._ui.dialogs.message_box_critical(
                "Display Error",
                f"Failed to display results:\n\n{exc}",
            )
            if self._display_exception_requires_terminal_cleanup(state):
                if isinstance(state.ctx, Mapping):
                    state.ctx = self._batch_context_owner.deactivate_if_active(state.ctx)
                state.batch_queue_done = True
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
        if not solver_provenance and isinstance(result.get("solver_provenance"), Mapping):
            solver_provenance = result.get("solver_provenance")
        mechanism = result.get("mechanism")
        base_species_count = self._base_species_count(result, mechanism=mechanism)
        mechanism_text = str(result.get("mechanism_text", ""))
        solver_config = result.get("solver_config", {})
        solver_config_map = solver_config if isinstance(solver_config, Mapping) else {}
        warnings = result.get("warnings") or []
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
            warnings=warnings if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) else (),
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
            solver_config=self._mechanism_resolution_solver_config(completion),
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
        context = self._explicit_cache_policy_context(state)
        if context is None:
            return
        cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
            context=context,
            cache_state=self._completion_policy_cache_state(),
            cache_key=state.cache_key,
        )
        self._cache_admin.publish_completion_cache_truth(
            is_preview=False,
            cache_key=state.cache_key,
            clear_active_cache_identity_state=cache_reconciliation.clear_active_cache_identity_state,
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
                completion.redraw_valid_set_ids, completion.has_redraw_subset = self._redraw_scope(state)
                return
            state.ctx = self._batch_context_owner.serialize_completion_policy_context(
                state.policy_context,
                base_context=state.ctx if isinstance(state.ctx, Mapping) else None,
            )
            completion.redraw_valid_set_ids, completion.has_redraw_subset = self._redraw_scope(state)

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
        owned_species = ()
        if isinstance(state.ctx, Mapping):
            owned_species = self._batch_context_owner.launch_owned_species_for_computed_result(
                state.ctx,
                set_id=state.batch_set_id,
            )
        materialized = self._materialize_completion_display(
            completion,
            required_owned_species=owned_species,
        )
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
            warnings=[
                dict(warning)
                for warning in completion.warnings
                if isinstance(warning, Mapping)
            ],
            completion_provenance=self.direct_completion_provenance_payload(completion),
            owned_species=owned_species,
            display_species=(
                materialized.display_species
                if isinstance(materialized, MaterializedDisplayResult)
                else ()
            ),
        )

    def record_in_flight_completion_display_entry(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> None:
        if state.is_preview or not state.cache_key or not isinstance(state.ctx, Mapping):
            return
        owned_species = self._batch_context_owner.launch_owned_species_for_computed_result(
            state.ctx,
            set_id=state.batch_set_id,
        )
        if not owned_species:
            state.ctx = self._batch_context_owner.record_completion_display_unavailable(
                state.ctx,
                set_id=state.batch_set_id,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )
            return
        materialized = self._materialize_completion_display(
            completion,
            required_owned_species=owned_species,
        )
        if materialized is None:
            state.ctx = self._batch_context_owner.record_completion_display_unavailable(
                state.ctx,
                set_id=state.batch_set_id,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )
            return
        entry = CompletionDisplayEntry(
            set_id=str(state.batch_set_id or ""),
            label=str(state.batch_set or state.batch_set_id or ""),
            t=completion.t,
            series=materialized.series,
            algebra_scalars=materialized.algebra_scalars,
            solver_provenance=completion.solver_provenance,
            mechanism_text=str(completion.mechanism_text or ""),
            solver_config=dict(completion.solver_config),
            warnings=tuple(
                dict(warning)
                for warning in completion.warnings
                if isinstance(warning, Mapping)
            ),
            completion_provenance=self.direct_completion_provenance_payload(completion),
            owned_species=materialized.owned_species,
            display_species=materialized.display_species,
        )
        state.ctx = self._batch_context_owner.record_completion_display_entry(
            state.ctx,
            set_id=state.batch_set_id,
            label=state.batch_set,
            entry=entry,
        )

    def _materialize_completion_display(
        self,
        completion: CompletionResultState,
        *,
        required_owned_species: Sequence[str],
    ) -> Optional[MaterializedDisplayResult]:
        return self._result_materialization_owner.materialize_completion_display_result(
            series=completion.series,
            finalized_species_names=completion.species_names,
            owned_species=required_owned_species,
            algebra_scalars=completion.algebra_scalars,
        )

    def _fresh_preview_display_entry(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> Optional[FreshPreviewDisplayEntry]:
        if not (state.cache_key and state.slider_triggered and state.is_preview):
            return None
        set_id = str(state.batch_set_id or "").strip()
        if not set_id:
            return None
        workspace_provenance: Mapping[str, Any] | None = None
        try:
            candidate = self._ui.batch.current_workspace_preview_identity_payload(set_id=set_id)
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping):
            workspace_provenance = dict(candidate)
        owned_species = self._base_owned_species_from_completion(completion)
        materialized = self._materialize_completion_display(
            completion,
            required_owned_species=owned_species,
        )
        if materialized is None:
            return None
        return FreshPreviewDisplayEntry(
            set_id=set_id,
            label=str(state.batch_set or set_id),
            t=completion.t,
            series=materialized.series,
            algebra_scalars=materialized.algebra_scalars,
            solver_provenance=completion.solver_provenance,
            completion_provenance=self.direct_completion_provenance_payload(completion),
            owned_species=materialized.owned_species,
            display_species=materialized.display_species,
            workspace_preview_provenance=workspace_provenance,
        )

    def _deferred_display_outcome(
        self,
        coverage: CompletedRunDisplayCoverage | None = None,
    ) -> SimulationCompletionDisplayOutcome:
        if coverage is None:
            return self._ui.results.publish_deferred_display_request()
        return self._ui.results.publish_deferred_display_request(
            affected_set_ids=self._completed_run_coverage_affected_set_ids(coverage),
            requested_show_set_ids=(
                tuple(str(set_id) for set_id in coverage.intent.requested_show_set_ids if str(set_id))
                if coverage.intent is not None
                else ()
            ),
            requested_labels_by_set_id=(
                {
                    str(set_id): str(label)
                    for set_id, label in dict(coverage.intent.labels_by_set_id or {}).items()
                    if str(set_id)
                }
                if coverage.intent is not None
                else {}
            ),
            unresolved_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.unresolved_intent_set_ids if str(set_id)
            ),
            missing_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.missing_set_ids if str(set_id)
            ),
            failed_intent_set_ids=tuple(
                str(set_id) for set_id in coverage.failed_intent_set_ids if str(set_id)
            ),
            semantic_unavailable_set_ids=tuple(
                str(set_id) for set_id in coverage.semantic_unavailable_set_ids if str(set_id)
            ),
        )

    @staticmethod
    def _completed_run_coverage_affected_set_ids(
        coverage: CompletedRunDisplayCoverage | None,
    ) -> tuple[str, ...]:
        if coverage is None:
            return ()
        for values in (
            coverage.unresolved_intent_set_ids,
            coverage.unavailable_set_ids,
            coverage.missing_set_ids,
        ):
            affected = tuple(str(set_id) for set_id in (values or ()) if str(set_id))
            if affected:
                return affected
        intent = coverage.intent
        if intent is not None:
            return tuple(str(set_id) for set_id in (intent.requested_show_set_ids or ()) if str(set_id))
        return ()

    def _completed_run_unavailable_outcome(
        self,
        coverage: CompletedRunDisplayCoverage | None,
    ) -> SimulationCompletionDisplayOutcome:
        cause = (
            coverage.cause
            if coverage is not None and isinstance(coverage.cause, DisplayTransitionCause)
            else DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE
        )
        return self.publish_completed_run_display_unavailable(
            cause=cause,
            affected_set_ids=self._completed_run_coverage_affected_set_ids(coverage),
            requested_show_set_ids=(
                tuple(str(set_id) for set_id in coverage.intent.requested_show_set_ids if str(set_id))
                if coverage is not None and coverage.intent is not None
                else ()
            ),
            requested_labels_by_set_id=(
                {
                    str(set_id): str(label)
                    for set_id, label in dict(coverage.intent.labels_by_set_id or {}).items()
                    if str(set_id)
                }
                if coverage is not None and coverage.intent is not None
                else {}
            ),
            unresolved_intent_set_ids=(
                tuple(str(set_id) for set_id in coverage.unresolved_intent_set_ids if str(set_id))
                if coverage is not None
                else ()
            ),
            missing_intent_set_ids=(
                tuple(str(set_id) for set_id in coverage.missing_set_ids if str(set_id))
                if coverage is not None
                else ()
            ),
            failed_intent_set_ids=(
                tuple(str(set_id) for set_id in coverage.failed_intent_set_ids if str(set_id))
                if coverage is not None
                else ()
            ),
            semantic_unavailable_set_ids=(
                tuple(str(set_id) for set_id in coverage.semantic_unavailable_set_ids if str(set_id))
                if coverage is not None
                else ()
            ),
        )

    def publish_display(
        self,
        completion: CompletionResultState,
        state: CompletionCallbackState,
    ) -> SimulationCompletionDisplayOutcome:
        requires_completed_run_transaction = self._requires_completed_run_display_transaction(state)
        completed_run_coverage = self._completed_run_display_coverage(state)
        completed_run_transaction = completed_run_coverage.transaction if completed_run_coverage is not None else None
        if state.cache_key and state.slider_triggered and not requires_completed_run_transaction:
            self._deps.queue_slider_plot_update(
                set_id=state.batch_set_id,
                cache_key=str(state.cache_key),
                request_id=state.request_id,
                run_id=state.run_id,
                slider_triggered=bool(state.slider_triggered),
                valid_set_ids=None,
                fresh_preview_entry=self._fresh_preview_display_entry(completion, state),
            )
            return self._deferred_display_outcome()

        if completed_run_transaction is not None:
            return self.publish_completed_run_display_transaction(completed_run_transaction)
        defer_in_progress = self._should_defer_in_progress_batch_display(state)
        if defer_in_progress:
            return self._deferred_display_outcome(completed_run_coverage)
        if requires_completed_run_transaction:
            return self._completed_run_unavailable_outcome(completed_run_coverage)

        owned_species = self._base_owned_species_from_completion(completion)
        materialized = self._materialize_completion_display(
            completion,
            required_owned_species=owned_species,
        )
        if materialized is None:
            return self._ui.results.publish_direct_completion_result(
                t=completion.t,
                series={},
                batch_set=state.batch_set,
                batch_set_id=state.batch_set_id,
                algebra_scalars=completion.algebra_scalars,
                solver_provenance=completion.solver_provenance,
                direct_completion_provenance=self.direct_completion_provenance_payload(completion),
                owned_species=owned_species,
                display_species=(),
            )
        outcome = self._ui.results.publish_direct_completion_result(
            t=completion.t,
            series=materialized.series,
            batch_set=state.batch_set,
            batch_set_id=state.batch_set_id,
            algebra_scalars=materialized.algebra_scalars,
            solver_provenance=completion.solver_provenance,
            direct_completion_provenance=self.direct_completion_provenance_payload(completion),
            owned_species=materialized.owned_species,
            display_species=materialized.display_species,
        )
        if isinstance(outcome, SimulationCompletionDisplayOutcome):
            return outcome
        raise RuntimeError("ResultsController returned an invalid direct display outcome")

    def _non_displayed_outcome_is_terminal(
        self,
        display_outcome: SimulationCompletionDisplayOutcome,
        state: CompletionCallbackState,
    ) -> bool:
        transition_outcome = display_outcome.transition_outcome
        if isinstance(transition_outcome, DisplayTransitionOutcome):
            if transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED:
                return False
            if transition_outcome.kind is DisplayTransitionOutcomeKind.DEFERRED:
                return False
            if (
                transition_outcome.cause
                is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE
            ):
                return False
            if transition_outcome.display_status in {
                DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            }:
                return False
            if state.cache_key and transition_outcome.cause in {
                DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                DisplayTransitionCause.INVALID_CACHE_ENTRY,
            }:
                return False
            return True
        return True

    def _display_exception_requires_terminal_cleanup(self, state: CompletionCallbackState) -> bool:
        if state.cache_key and state.slider_triggered:
            return False
        try:
            return not self._should_defer_in_progress_batch_display(state)
        except Exception:
            return True

    def _should_defer_in_progress_batch_display(self, state: CompletionCallbackState) -> bool:
        if (not state.cache_key) or state.is_preview or not isinstance(state.ctx, Mapping):
            return False
        completion_state = self._batch_context_owner.completion_state(state.ctx)
        if completion_state is None or not completion_state.active:
            return False
        queue_ids = tuple(str(set_id) for set_id in (completion_state.queue_ids or ()) if str(set_id))
        total = max(1, int(completion_state.total or len(queue_ids) or 1))
        if total <= 1:
            return False
        current_set_id = str(state.batch_set_id or "").strip()
        if completion_state.parallel:
            completed = {str(set_id) for set_id in completion_state.completed_set_ids if str(set_id)}
            if current_set_id:
                completed.add(current_set_id)
            return len(completed) < total
        if state.shutdown_requested:
            return False
        pos = max(0, int(completion_state.pos or 0))
        if queue_ids and 0 <= pos < len(queue_ids):
            expected_set_id = str(queue_ids[pos])
            if current_set_id and current_set_id != expected_set_id:
                return False
            return (pos + 1) < len(queue_ids)
        return (pos + 1) < total

    def _requires_completed_run_display_transaction(self, state: CompletionCallbackState) -> bool:
        if (not state.cache_key) or state.is_preview or not isinstance(state.ctx, Mapping):
            return False
        completion_state = self._batch_context_owner.completion_state(state.ctx)
        if completion_state is None or not completion_state.active:
            return False
        queue_ids = tuple(str(set_id) for set_id in (completion_state.queue_ids or ()) if str(set_id))
        total = max(1, int(completion_state.total or len(queue_ids) or 1))
        return bool(queue_ids) and total >= 1

    def _completed_run_display_coverage(
        self,
        state: CompletionCallbackState,
    ) -> CompletedRunDisplayCoverage | None:
        if not self._requires_completed_run_display_transaction(state):
            return None
        return self._batch_context_owner.completed_run_display_coverage(state.ctx)

    def publish_completed_run_display_transaction(
        self,
        transaction: CompletedRunDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome:
        outcome = self._ui.results.publish_completed_run_display_transaction(transaction)
        if not isinstance(outcome, SimulationCompletionDisplayOutcome):
            raise RuntimeError("ResultsController returned an invalid completed-run display outcome")
        return outcome

    def publish_completed_run_display_unavailable(
        self,
        *,
        cause: DisplayTransitionCause,
        affected_set_ids: Sequence[str],
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> SimulationCompletionDisplayOutcome:
        if not isinstance(cause, DisplayTransitionCause):
            raise TypeError("Completed-run display unavailable requires DisplayTransitionCause")
        outcome = self._ui.results.publish_completed_run_display_unavailable(
            cause=cause,
            affected_set_ids=affected_set_ids,
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            attempted_display_set_ids=attempted_display_set_ids,
            unresolved_intent_set_ids=unresolved_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
        )
        if not isinstance(outcome, SimulationCompletionDisplayOutcome):
            raise RuntimeError("ResultsController returned an invalid unavailable display outcome")
        return outcome

    def direct_completion_provenance_payload(
        self,
        completion: CompletionResultState,
    ) -> Optional[Dict[str, Any]]:
        launch_provenance = (
            completion.solver_provenance.get("launch_provenance")
            if isinstance(completion.solver_provenance, Mapping)
            else None
        )
        launch_provenance = launch_provenance if isinstance(launch_provenance, Mapping) else {}
        temperature_used = None
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
        if launch_provenance.get("temperature_K") is not None:
            temperature_used = float(launch_provenance["temperature_K"])
        if temperature_used is None:
            raise ValueError("Completion provenance requires captured temperature_K.")

        if launch_provenance.get("simulation_time") is not None:
            sim_time_prov: float | str = float(launch_provenance["simulation_time"])
        else:
            try:
                sim_time_prov = float(completion.t[-1])
            except Exception as exc:
                raise ValueError("Completion provenance requires captured simulation_time.") from exc

        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

        solver_label = str(
            completion.solver_config.get("solver_label")
            or completion.solver_config.get("solver")
            or DEFAULT_SOLVER_NAME
        ).strip() or DEFAULT_SOLVER_NAME
        solver_method, solver_warning = normalize_solver_name(
            str(completion.solver_config.get("solver") or solver_label)
        )
        return {
            "mechanism_text": completion.mechanism_text,
            "solver_method": str(solver_method),
            "solver_label": str(solver_label),
            "solver_warning": (str(solver_warning) if solver_warning else None),
            "solver_config": {
                "rtol": completion.solver_config.get("rtol", 1e-6),
                "atol": completion.solver_config.get("atol", 1e-12),
            },
            "temperature_K": float(temperature_used),
            "temperature_source": (
                str(launch_provenance.get("temperature_source"))
                if launch_provenance.get("temperature_source") is not None
                else "result"
            ),
            "energy_unit": energy_unit_used,
            "energy_mode": bool(completion.energy_mode),
            "simulation_time": sim_time_prov,
            "num_points_requested": (
                int(launch_provenance["num_points_requested"])
                if launch_provenance.get("num_points_requested") is not None
                else int((completion.solver_config.get("grid") or {}).get("N") or len(completion.t))
            ),
            "species_names": list(completion.species_names),
            "t": completion.t,
            "series": completion.series,
            "algebra_scalars": completion.algebra_scalars,
            "solver_provenance": completion.solver_provenance,
            "warnings": completion.warnings,
        }

    @staticmethod
    def _base_owned_species_from_completion(
        completion: CompletionResultState,
    ) -> tuple[str, ...]:
        available_series = {str(name) for name in dict(completion.series or {}) if str(name)}
        base_species_count = completion.base_species_count
        if isinstance(base_species_count, int) and base_species_count >= 0:
            owned_source = tuple(completion.species_names or ())[:base_species_count]
        else:
            owned_source = ()
        return tuple(
            str(name)
            for name in owned_source
            if str(name) and (not available_series or str(name) in available_series)
        )

    @staticmethod
    def _mechanism_resolution_solver_config(
        completion: CompletionResultState,
    ) -> Mapping[str, Any]:
        solver_config = dict(completion.solver_config or {})
        if solver_config.get("temperature_K") is not None:
            return solver_config
        launch_provenance = (
            completion.solver_provenance.get("launch_provenance")
            if isinstance(completion.solver_provenance, Mapping)
            else None
        )
        if not isinstance(launch_provenance, Mapping):
            return solver_config
        if launch_provenance.get("temperature_K") is None:
            return solver_config
        solver_config["temperature_K"] = launch_provenance["temperature_K"]
        return solver_config

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
        *,
        display_outcome: SimulationCompletionDisplayOutcome | None = None,
    ) -> None:
        if (not state.is_preview) and isinstance(state.ctx, dict):
            state.ctx = self._deps.finalize_explicit_batch_dirty_reset(
                state.ctx,
                mechanism=completion.mechanism,
                species_names=completion.species_names,
            )
        has_completed_run_display_intent = (
            isinstance(state.ctx, Mapping)
            and isinstance(
                state.ctx.get("completed_run_display_intent"),
                CompletedRunDisplayIntent,
            )
        )
        if state.slider_triggered and (state.is_preview or not has_completed_run_display_intent):
            self._deps.flush_slider_plot_updates(
                force=False,
                cache_key=state.cache_key,
                request_id=state.request_id,
                run_id=state.run_id,
        )
        summary = self._batch_context_owner.completion_summary(state.ctx) if isinstance(state.ctx, Mapping) else None
        failed_set_ids = summary.failed_set_ids if summary is not None else ()
        completion_effect = self._lifecycle_effect_owner.completion_status_effect(
            species_count=len(completion.species_names),
            point_count=len(completion.t),
            failed_set_ids=failed_set_ids,
            is_preview=bool(state.is_preview),
        )
        if isinstance(display_outcome.transition_outcome, DisplayTransitionOutcome):
            self._deps.clear_pending_progress_status()
            completion_effect = replace(completion_effect, status_text=None)
        self._deps.apply_lifecycle_effects(completion_effect)
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
        context = self._explicit_cache_policy_context(state)
        if context is None:
            return None, False
        cache_reconciliation = self._completion_policy.build_explicit_cache_reconciliation(
            context=context,
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

    def _explicit_cache_policy_context(self, state: CompletionCallbackState) -> CompletionPolicyContext | None:
        context = self._batch_context_owner.completion_publication_policy_context(
            callback_context=state.ctx if isinstance(state.ctx, Mapping) else None,
            policy_context=state.policy_context,
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
