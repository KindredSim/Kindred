from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from kindred.core.batch_parallel import batch_mechanism_signature
from kindred.core.batch_initial_conditions import (
    migrate_reaction_dsl_initial_concentration_sets,
    strip_reaction_dsl_initial_concentrations,
)
from kindred.core.simulation_identity import (
    SimulationScopeIdentity,
    canonical_initials_fingerprint,
    coerce_simulation_identity,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy
from kindred.gui.controllers.batch_run_context_owner import BatchRunStartRequest
from kindred.gui.controllers.batch_dispatch_plan import BatchSetDispatchInput, build_batch_set_dispatch_plan
from kindred.gui.project_schema import PROJECT_DEFAULTS

logger = logging.getLogger(__name__)

__all__ = [
    "RunDispatchContext",
    "RunMechanismContext",
    "RunSolverContext",
    "SimulationRunPreparationDependencies",
    "SimulationRunPreparationOwner",
    "SimulationRunPreparationPorts",
    "build_fast_preview_solver_grid_context",
]


@dataclass
class RunMechanismContext:
    batch_rows: List[int]
    queue_names: List[str]
    queue_ids: List[str]
    has_slider_overrides: bool
    primary: object | None
    primary_set_id: str | None
    owner_full_dsl: str
    full_dsl: str
    pending_init_seed: Dict[str, Dict[str, float]]
    pending_init_rewrite: Optional[str]
    pending_init_applied: bool
    preview_owner_epoch: int | None


@dataclass
class RunSolverContext:
    solver_config: Dict[str, Any]
    t_end: float
    prepared_payload: Dict[str, Any] | None
    prepared_payload_by_set_id: Dict[str, Dict[str, Any]]
    execution_prepared_payload_by_set_id: Dict[str, Dict[str, Any]]
    owner_parameter_names_by_set_id: Dict[str, List[str]]


@dataclass
class RunDispatchContext:
    simulation_plan_by_set_id: Dict[str, Dict[str, Any]]
    mechanism_text_by_set_id: Dict[str, str]
    mechanism_signature_by_set_id: Dict[str, str]
    simulation_identity_by_set_id: Dict[str, Dict[str, Any]]
    preview_batch_cache_token_by_set_id: Dict[str, str]
    scope_identity: SimulationScopeIdentity
    cache_key: str


@dataclass(frozen=True)
class RunStartContext:
    request: BatchRunStartRequest
    parallel_mode: bool
    effective_workers: int
    run_id: int | None
    run_sequence_id: int


@dataclass(frozen=True)
class SimulationRunPreparationPorts:
    batch: Any
    dialogs: Any
    mechanism: Any
    mechanism_helpers: Any
    run_ui: Any
    slider: Any
    solver: Any


@dataclass(frozen=True)
class SimulationRunPreparationDependencies:
    claim_preview_ownership: Callable[..., Any]
    clear_preview_ownership: Callable[[], None]
    invalidate_preserved_pending_init_results_after_failed_run: Callable[..., None]
    clear_failed_fast_preview_ownership: Callable[[], None]
    clear_slider_triggered_preflight_state: Callable[..., None]
    requeue_preserved_pending_slider_replay_after_preflight_abort: Callable[[], None]
    record_nonfatal_exception: Callable[[str, BaseException], None]
    set_simulation_running: Callable[[bool], None]
    set_slider_simulation_active: Callable[[bool], None]
    sync_batch_species_columns_for_run: Callable[..., None]
    slider_runtime_parameter_names: Callable[..., Sequence[str]]
    simulation_identity_for_set: Callable[..., Any]
    request_mechanism_text_for_set: Callable[..., str]
    resolved_initials_for_batch_row: Callable[..., Dict[str, Any]]
    slider_execution_parameter_values: Callable[..., Dict[str, Any]]
    preview_contained_owner_identity: Callable[..., Dict[str, Any]]
    ordinary_contained_owner_identity: Callable[..., Dict[str, Any]]
    record_run_cache_key: Callable[..., None]


def build_fast_preview_solver_grid_context(
    *,
    initial_solver_name: Optional[str],
    num_points: int,
    fast_mode: bool,
    slider_points_override: Optional[int],
    slider_solver_override: Optional[str],
    slider_drag_active: bool,
    last_slider_change_name: str,
) -> Dict[str, Any]:
    from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

    solver_label = str(initial_solver_name or DEFAULT_SOLVER_NAME).strip() or DEFAULT_SOLVER_NAME
    solver, solver_warning = normalize_solver_name(solver_label)
    n_points = int(num_points)

    if fast_mode:
        if slider_points_override is not None:
            n_points = max(50, int(slider_points_override))
        else:
            n_points = max(50, n_points)
        if slider_solver_override is not None:
            solver_label = str(slider_solver_override).strip() or solver_label
            solver, solver_warning = normalize_solver_name(solver_label)

    preview_mode = bool(
        fast_mode
        and slider_drag_active
        and isinstance(last_slider_change_name, str)
        and last_slider_change_name.startswith("Keq")
        and last_slider_change_name[3:].isdigit()
    )
    if preview_mode:
        n_points = min(int(n_points), 120)

    return {
        "solver": str(solver),
        "solver_label": str(solver_label),
        "solver_warning": str(solver_warning) if solver_warning else None,
        "grid": {"N": int(n_points)},
    }


def build_run_start_context(
    *,
    request_id: int,
    current_run_sequence_id: int,
    runtime_input_epoch: int,
    runtime_input_global_epoch: int,
    runtime_input_set_epoch_by_set_id: Mapping[str, Any],
    fast_mode: bool,
    reuse_parallel_lane_pool: bool,
    effective_workers: int,
    mechanism_context: RunMechanismContext,
    solver_context: RunSolverContext,
    dispatch_context: RunDispatchContext,
    run_start_cache_decision: Any,
    dirty_reset_tracking: Any,
) -> RunStartContext:
    queue_ids = list(mechanism_context.queue_ids)
    parallel_mode = bool(int(effective_workers) > 1 and len(queue_ids) > 1)
    run_sequence_id = int(current_run_sequence_id)
    run_id = None
    if parallel_mode:
        run_sequence_id += 1
        run_id = int(run_sequence_id)

    retain_prepared_payloads_in_context = not (bool(parallel_mode) and not bool(fast_mode))
    primary_simulation_plan = None
    if not bool(fast_mode):
        if mechanism_context.primary_set_id:
            primary_simulation_plan = dispatch_context.simulation_plan_by_set_id.get(
                str(mechanism_context.primary_set_id)
            )
        if primary_simulation_plan is None and dispatch_context.simulation_plan_by_set_id:
            primary_simulation_plan = dict(next(iter(dispatch_context.simulation_plan_by_set_id.values())))

    primary_mechanism_signature = (
        str(dispatch_context.mechanism_signature_by_set_id.get(str(mechanism_context.primary_set_id or "")) or "")
        if mechanism_context.primary_set_id
        else ""
    ) or batch_mechanism_signature(
        simulation_identity=(
            coerce_simulation_identity(
                dispatch_context.simulation_identity_by_set_id.get(str(mechanism_context.primary_set_id or ""))
            )
            if mechanism_context.primary_set_id
            else None
        ),
    )

    request = BatchRunStartRequest(
        request_id=int(request_id),
        run_id=run_id,
        runtime_input_epoch=int(runtime_input_epoch),
        runtime_input_global_epoch=int(runtime_input_global_epoch),
        runtime_input_set_epoch_by_set_id=runtime_input_set_epoch_by_set_id,
        fast_mode=bool(fast_mode),
        reuse_parallel_lane_pool=bool(reuse_parallel_lane_pool),
        parallel=bool(parallel_mode),
        effective_workers=int(effective_workers),
        retain_prepared_payloads_in_context=bool(retain_prepared_payloads_in_context),
        prepared_payload=solver_context.prepared_payload,
        prepared_payload_by_set_id=solver_context.prepared_payload_by_set_id,
        primary_simulation_plan=primary_simulation_plan if isinstance(primary_simulation_plan, Mapping) else None,
        simulation_plan_by_set_id=dispatch_context.simulation_plan_by_set_id,
        cache_key=dispatch_context.cache_key,
        scope_identity=dispatch_context.scope_identity.to_payload(),
        full_dsl=mechanism_context.full_dsl,
        mechanism_text_by_set_id=dispatch_context.mechanism_text_by_set_id,
        mechanism_signature=primary_mechanism_signature,
        mechanism_signature_by_set_id=dispatch_context.mechanism_signature_by_set_id,
        simulation_identity_by_set_id=dispatch_context.simulation_identity_by_set_id,
        solver_config=solver_context.solver_config,
        t_end=float(solver_context.t_end),
        rows=list(mechanism_context.batch_rows),
        queue_ids=list(queue_ids),
        queue_names=list(mechanism_context.queue_names),
        pending_workspace_reset_set_ids=list(dirty_reset_tracking.pending_workspace_reset_set_ids),
        pending_dirty_reset_generation_by_set_id=dict(dirty_reset_tracking.pending_dirty_reset_generation_by_set_id),
        primary_set_id=mechanism_context.primary,
        pending_init_seed=mechanism_context.pending_init_seed,
        pending_init_rewrite=mechanism_context.pending_init_rewrite,
        pending_init_applied=bool(mechanism_context.pending_init_applied),
        explicit_cache_preview_token=None,
        explicit_cache_preview_scope_set_ids=run_start_cache_decision.explicit_preview_scope_set_ids,
        explicit_cache_valid_set_ids=run_start_cache_decision.explicit_cache_valid_set_ids,
        explicit_cache_invalidated_set_ids=run_start_cache_decision.explicit_cache_invalidated_set_ids,
        preview_scope_set_ids=run_start_cache_decision.preview_scope_set_ids,
        preview_owner_epoch=mechanism_context.preview_owner_epoch,
        preview_batch_cache_token_by_set_id=dispatch_context.preview_batch_cache_token_by_set_id,
    )
    return RunStartContext(
        request=request,
        parallel_mode=bool(parallel_mode),
        effective_workers=int(effective_workers),
        run_id=run_id,
        run_sequence_id=int(run_sequence_id),
    )


class SimulationRunMechanismPreparationOwner:
    def __init__(
        self,
        *,
        ports: SimulationRunPreparationPorts,
        dependencies: SimulationRunPreparationDependencies,
    ) -> None:
        self._ports = ports
        self._deps = dependencies

    def build_mechanism_context_or_abort(
        self,
        *,
        fast_mode: bool,
        request_id: int,
        batch_rows: Sequence[int],
        runtime_readiness_only: bool,
    ) -> RunMechanismContext | None:
        any_slider_workspace = bool(self._ports.mechanism.has_slider_overrides())
        has_slider_overrides = bool(fast_mode) and bool(any_slider_workspace)
        primary = self._ports.batch.batch_preferred_primary_set_id(batch_rows)
        primary_set_id = str(primary) if primary is not None else None

        owner_reactions_text_raw = self._ports.mechanism.mechanism_reactions_text_raw()
        owner_state_network_dsl_raw = self._ports.mechanism.mechanism_state_network_dsl_raw()
        reactions_text_raw = owner_reactions_text_raw
        if has_slider_overrides:
            reactions_text_raw = self._ports.mechanism.apply_overrides_to_text(
                reactions_text_raw,
                set_id=primary_set_id,
            )

        pending_init_seed: Dict[str, Dict[str, float]] = {}
        pending_init_rewrite: Optional[str] = None
        pending_init_applied = False
        migrated = reactions_text_raw
        try:
            pending_init_seed, migrated = migrate_reaction_dsl_initial_concentration_sets(
                reactions_text_raw,
                default_set_name="set1",
            )
            if pending_init_seed:
                pending_init_rewrite = migrated
        except Exception:
            pending_init_seed = {}
            pending_init_rewrite = None
            migrated = reactions_text_raw

        rows = list(batch_rows)
        if (
            (not bool(fast_mode))
            and pending_init_seed
            and pending_init_rewrite
            and not bool(runtime_readiness_only)
        ):
            try:
                pending_init_applied = bool(
                    self._ports.mechanism_helpers.apply_pending_init_migration(
                        seed_sets=dict(pending_init_seed),
                        rewrite=str(pending_init_rewrite),
                    )
                )
            except Exception:
                pending_init_applied = False
            if pending_init_applied:
                reactions_text_raw = str(pending_init_rewrite)
                imported_names = [str(name) for name in pending_init_seed.keys() if str(name)]
                materialized_names = list(self._ports.batch.batch_store_set_names())
                materialized_rows: List[int] = []
                for imported_name in imported_names:
                    try:
                        row_idx = materialized_names.index(str(imported_name))
                    except ValueError:
                        continue
                    materialized_rows.append(int(row_idx))
                if materialized_rows:
                    rows = materialized_rows
                    primary = self._ports.batch.batch_preferred_primary_set_id(rows)
                    primary_set_id = str(primary) if primary is not None else None

        names = list(self._ports.batch.batch_store_set_names())
        queue_names = [str(names[r]) for r in rows if 0 <= int(r) < len(names)]
        queue_ids = [
            str(self._ports.batch.batch_set_id_for_row(int(r)) or str(names[int(r)]))
            for r in rows
            if 0 <= int(r) < len(names)
        ]
        preview_owner_epoch = None
        if bool(fast_mode) and not bool(runtime_readiness_only):
            preview_ownership = self._deps.claim_preview_ownership(
                request_id=int(request_id),
                target_set_ids=queue_ids,
            )
            preview_owner_epoch = int(preview_ownership.epoch)
        elif (not bool(fast_mode)) and (not bool(runtime_readiness_only)):
            self._deps.clear_preview_ownership()

        reactions_text = strip_reaction_dsl_initial_concentrations(
            reactions_text_raw if pending_init_applied else migrated
        )
        state_network_dsl = owner_state_network_dsl_raw
        if has_slider_overrides:
            state_network_dsl = self._ports.mechanism.apply_overrides_to_state_network_dsl(
                state_network_dsl,
                set_id=primary_set_id,
            )
        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        owner_reactions_text = strip_reaction_dsl_initial_concentrations(owner_reactions_text_raw)
        owner_full_dsl = owner_reactions_text
        if owner_state_network_dsl_raw.strip():
            owner_full_dsl += "\n\n# State Network\n" + owner_state_network_dsl_raw

        if not full_dsl.strip():
            if bool(runtime_readiness_only):
                return None
            self._deps.invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(pending_init_applied)
            )
            self._ports.dialogs.message_box_warning(
                "No Mechanism",
                "Please define reactions or state network in the Mechanism editor first.",
            )
            if bool(fast_mode):
                self._deps.clear_failed_fast_preview_ownership()
            self._ports.run_ui.set_status_text("Ready")
            self._deps.set_simulation_running(False)
            self._ports.run_ui.set_run_button_enabled(True)
            self._ports.run_ui.set_stop_button_enabled(False)
            self._ports.batch.update_batch_row_controls_state()
            self._deps.clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            self._deps.requeue_preserved_pending_slider_replay_after_preflight_abort()
            return None

        return RunMechanismContext(
            batch_rows=rows,
            queue_names=queue_names,
            queue_ids=queue_ids,
            has_slider_overrides=bool(has_slider_overrides),
            primary=primary,
            primary_set_id=primary_set_id,
            owner_full_dsl=owner_full_dsl,
            full_dsl=full_dsl,
            pending_init_seed=pending_init_seed,
            pending_init_rewrite=pending_init_rewrite,
            pending_init_applied=bool(pending_init_applied),
            preview_owner_epoch=preview_owner_epoch,
        )


class SimulationRunSolverPreparationOwner:
    def __init__(
        self,
        *,
        ports: SimulationRunPreparationPorts,
        dependencies: SimulationRunPreparationDependencies,
    ) -> None:
        self._ports = ports
        self._deps = dependencies

    def build_solver_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
    ) -> RunSolverContext | None:
        solver_grid_context = build_fast_preview_solver_grid_context(
            initial_solver_name=self._ports.solver.initial_solver_name(),
            num_points=int(self._ports.solver.num_points_spinbox_value()),
            fast_mode=bool(fast_mode),
            slider_points_override=self._ports.mechanism.mechanism_slider_points_value(),
            slider_solver_override=self._ports.mechanism.mechanism_slider_solver_value(),
            slider_drag_active=bool(self._ports.slider.slider_drag_active()),
            last_slider_change_name=str(self._ports.slider.last_slider_change_name() or ""),
        )
        solver_label = str(solver_grid_context.get("solver_label") or "")
        solver = str(solver_grid_context.get("solver") or "")
        solver_warning = solver_grid_context.get("solver_warning")
        if solver_warning and not bool(runtime_readiness_only):
            logger.warning("Solver normalization: %s (requested=%r)", solver_warning, solver_label)
            try:
                self._ports.run_ui.set_status_text(str(solver_warning))
            except Exception:
                pass
        rtol = self._ports.solver.initial_rtol() or 1e-6
        atol = self._ports.solver.initial_atol() or 1e-12
        temperature_K = float(self._ports.solver.temperature_spinbox_value())
        T_override = self._ports.solver.dsl_global_temperature_K(mechanism_context.full_dsl)
        if T_override is not None:
            temperature_K = float(T_override)

        prepared_payload: Optional[Dict[str, Any]] = None
        prepared_payload_by_set_id: Dict[str, Dict[str, Any]] = {}
        execution_prepared_payload_by_set_id: Dict[str, Dict[str, Any]] = {}
        owner_parameter_names_by_set_id: Dict[str, List[str]] = {}
        slider_runtime = None
        if bool(fast_mode):
            target_runtime_set_ids = list(mechanism_context.queue_ids)
            if (not target_runtime_set_ids) and mechanism_context.primary_set_id:
                target_runtime_set_ids = [str(mechanism_context.primary_set_id)]
            for set_id in target_runtime_set_ids:
                runtime_parameter_names = self._deps.slider_runtime_parameter_names(set_id=str(set_id))
                owner_parameter_names_by_set_id[str(set_id)] = list(runtime_parameter_names)
            if mechanism_context.primary_set_id:
                prepared_payload = prepared_payload_by_set_id.get(str(mechanism_context.primary_set_id))
            if prepared_payload is None and prepared_payload_by_set_id:
                prepared_payload = dict(next(iter(prepared_payload_by_set_id.values())))

        if (not bool(runtime_readiness_only)) and (
            (not bool(fast_mode)) or (not list(self._ports.batch.batch_store_visible_species()))
        ):
            self._deps.sync_batch_species_columns_for_run(
                fast_mode=bool(fast_mode),
                slider_runtime=slider_runtime,
                full_dsl=mechanism_context.full_dsl,
                temperature_K=float(temperature_K),
            )

        n_points = int((solver_grid_context.get("grid") or {}).get("N") or 0)
        if fast_mode:
            logger.debug("Fast mode: using %s points and %s solver for slider update", n_points, solver)

        solver_config = {
            "solver": solver,
            "solver_label": solver_label,
            "solver_warning": str(solver_warning) if solver_warning else None,
            "rtol": rtol,
            "atol": atol,
            "grid": {"N": n_points},
            "temperature_K": temperature_K,
            "use_sparse_jacobian": bool(self._ports.solver.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(self._ports.solver.wegscheider_cyclicity_enabled()),
        }
        try:
            t_end = float(self._ports.solver.parse_sim_time_seconds())
        except ValueError as exc:
            if bool(runtime_readiness_only):
                return None
            self._deps.invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(mechanism_context.pending_init_applied)
            )
            self._ports.dialogs.message_box_warning("Invalid t_end", f"Fix t_end before running:\n\n{exc}")
            if bool(fast_mode):
                self._deps.clear_failed_fast_preview_ownership()
            self._deps.set_simulation_running(False)
            try:
                self._ports.run_ui.set_run_button_enabled(True)
            except Exception as ui_exc:
                self._deps.record_nonfatal_exception("Failed to re-enable Run button after invalid t_end", ui_exc)
            try:
                self._ports.run_ui.set_stop_button_enabled(False)
            except Exception as ui_exc:
                self._deps.record_nonfatal_exception("Failed to disable Stop button after invalid t_end", ui_exc)
            self._deps.set_slider_simulation_active(False)
            self._deps.clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            if not bool(fast_mode):
                self._deps.requeue_preserved_pending_slider_replay_after_preflight_abort()
            return None

        return RunSolverContext(
            solver_config=solver_config,
            t_end=float(t_end),
            prepared_payload=prepared_payload if isinstance(prepared_payload, dict) else None,
            prepared_payload_by_set_id=prepared_payload_by_set_id,
            execution_prepared_payload_by_set_id=execution_prepared_payload_by_set_id,
            owner_parameter_names_by_set_id=owner_parameter_names_by_set_id,
        )


class SimulationRunDispatchPreparationOwner:
    def __init__(
        self,
        *,
        ports: SimulationRunPreparationPorts,
        dependencies: SimulationRunPreparationDependencies,
    ) -> None:
        self._ports = ports
        self._deps = dependencies

    def build_dispatch_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
    ) -> RunDispatchContext | None:
        simulation_plan_by_set_id: Dict[str, Dict[str, Any]] = {}
        mechanism_text_by_set_id: Dict[str, str] = {}
        mechanism_signature_by_set_id: Dict[str, str] = {}
        preview_batch_cache_token_by_set_id: Dict[str, str] = {}
        for index, set_id in enumerate(mechanism_context.queue_ids):
            token = ""
            if bool(fast_mode) and index < len(mechanism_context.batch_rows):
                try:
                    token = self._ports.slider.preview_batch_cache_token([int(mechanism_context.batch_rows[index])])
                except Exception:
                    token = ""
            preview_batch_cache_token_by_set_id[str(set_id)] = str(token or "")

        simulation_identity_by_set_id: Dict[str, Dict[str, Any]] = {}
        initials_by_set_id: Dict[str, Dict[str, Any]] = {}
        parameter_overrides_by_set_id: Dict[str, Dict[str, Any]] = {}
        intervention_schedule_by_set_id: Dict[str, Dict[str, Any]] = {}
        contained_owner_identity_by_set_id: Dict[str, Dict[str, Any]] = {}

        def _abort_invalid_intervention_schedule(set_id_s: str, exc: BaseException) -> None:
            if bool(runtime_readiness_only):
                raise ValueError(f"Invalid intervention schedule for set {set_id_s!r}: {exc}") from exc
            self._deps.invalidate_preserved_pending_init_results_after_failed_run(
                pending_init_applied=bool(mechanism_context.pending_init_applied)
            )
            self._deps.record_nonfatal_exception("Failed to parse intervention schedule for run plan", exc)
            self._ports.dialogs.message_box_warning(
                "Invalid Intervention Schedule",
                f"Fix intervention schedule directives before running:\n\n{exc}",
            )
            if bool(fast_mode):
                self._deps.clear_failed_fast_preview_ownership()
            self._deps.set_simulation_running(False)
            self._ports.run_ui.set_run_button_enabled(True)
            self._ports.run_ui.set_stop_button_enabled(False)
            self._deps.set_slider_simulation_active(False)
            self._deps.clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            if not bool(fast_mode):
                self._deps.requeue_preserved_pending_slider_replay_after_preflight_abort()

        def _submitted_intervention_schedule_from_text(text: str):
            from kindred.core.intervention_schedule import (
                coerce_intervention_schedule,
                normalized_intervention_schedule_payload,
                parse_intervention_schedule_from_dsl,
            )

            intervention_schedule = parse_intervention_schedule_from_dsl(str(text or ""))
            if intervention_schedule is None:
                return None
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.simulator.parameter_namespace import build_namespace_from_mechanism

            mechanism = parse_dsl_to_mechanism(str(text or ""), initials={})
            namespace = build_namespace_from_mechanism(mechanism)
            payload = normalized_intervention_schedule_payload(
                intervention_schedule,
                mechanism_namespace=namespace,
            )
            return coerce_intervention_schedule(payload)

        for index, set_id in enumerate(mechanism_context.queue_ids):
            set_id_s = str(set_id)
            if index >= len(mechanism_context.batch_rows):
                try:
                    intervention_schedule = _submitted_intervention_schedule_from_text(
                        str(mechanism_context.owner_full_dsl or "")
                    )
                    identity = self._deps.simulation_identity_for_set(
                        set_id=set_id_s,
                        solver_config=solver_context.solver_config,
                        t_end=float(solver_context.t_end),
                        preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(set_id_s, ""),
                        intervention_schedule_fingerprint=(
                            "" if intervention_schedule is None else str(intervention_schedule.fingerprint or "")
                        ),
                        fast_mode=bool(fast_mode),
                    )
                except Exception as exc:
                    _abort_invalid_intervention_schedule(set_id_s, exc)
                    return None
                identity_payload = identity.to_payload()
                if intervention_schedule is not None:
                    intervention_schedule_by_set_id[set_id_s] = intervention_schedule.to_payload()
                simulation_identity_by_set_id[set_id_s] = identity_payload
                continue
            row = int(mechanism_context.batch_rows[index])
            set_name = (
                str(mechanism_context.queue_names[index])
                if index < len(mechanism_context.queue_names)
                else str(set_id)
            )
            request_mechanism_text = self._deps.request_mechanism_text_for_set(
                set_id=set_id_s,
                has_slider_overrides=mechanism_context.has_slider_overrides,
            )
            mechanism_text_by_set_id[set_id_s] = str(request_mechanism_text)
            try:
                intervention_schedule = _submitted_intervention_schedule_from_text(str(request_mechanism_text))
            except Exception as exc:
                _abort_invalid_intervention_schedule(set_id_s, exc)
                return None
            if intervention_schedule is not None:
                intervention_schedule_by_set_id[set_id_s] = intervention_schedule.to_payload()
            prepared_execution_payload = solver_context.execution_prepared_payload_by_set_id.get(set_id_s)
            try:
                initials_dict = self._deps.resolved_initials_for_batch_row(
                    row=row,
                    set_name=set_name,
                    pending_init_seed=mechanism_context.pending_init_seed,
                    pending_init_applied=False,
                    include_preview_initials=bool(fast_mode),
                )
            except Exception as exc:
                if bool(runtime_readiness_only):
                    return None
                if not bool(fast_mode):
                    self._deps.invalidate_preserved_pending_init_results_after_failed_run(
                        pending_init_applied=bool(mechanism_context.pending_init_applied)
                    )
                self._ports.dialogs.message_box_warning(
                    "Invalid Initial Conditions",
                    f"Set '{set_name}' has invalid initial conditions:\n\n{exc}",
                )
                if bool(fast_mode):
                    self._deps.clear_failed_fast_preview_ownership()
                self._deps.set_simulation_running(False)
                self._ports.run_ui.set_run_button_enabled(True)
                self._ports.run_ui.set_stop_button_enabled(False)
                self._deps.set_slider_simulation_active(False)
                if bool(fast_mode):
                    self._deps.clear_slider_triggered_preflight_state(fast_mode=True)
                else:
                    self._deps.requeue_preserved_pending_slider_replay_after_preflight_abort()
                return None

            try:
                identity = self._deps.simulation_identity_for_set(
                    set_id=set_id_s,
                    solver_config=solver_context.solver_config,
                    t_end=float(solver_context.t_end),
                    canonical_initials_fingerprint=canonical_initials_fingerprint(initials_dict),
                    preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(set_id_s, ""),
                    intervention_schedule_fingerprint=(
                        "" if intervention_schedule is None else str(intervention_schedule.fingerprint or "")
                    ),
                    fast_mode=bool(fast_mode),
                )
            except Exception as exc:
                _abort_invalid_intervention_schedule(set_id_s, exc)
                return None
            identity_payload = identity.to_payload()
            simulation_identity_by_set_id[set_id_s] = identity_payload
            initials_by_set_id[set_id_s] = dict(initials_dict)
            if bool(fast_mode):
                parameter_overrides_by_set_id[set_id_s] = self._deps.slider_execution_parameter_values(
                    set_id=set_id_s
                )
                if isinstance(prepared_execution_payload, dict):
                    mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                        simulation_identity=identity,
                    )
                else:
                    mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                        mechanism_text=str(request_mechanism_text),
                        temperature_K=float(solver_context.solver_config.get("temperature_K") or 298.15),
                        use_sparse_jacobian=bool(
                            solver_context.solver_config.get(
                                "use_sparse_jacobian",
                                PROJECT_DEFAULTS["use_sparse_jacobian"],
                            )
                        ),
                        wegscheider_cyclicity_enabled=bool(
                            solver_context.solver_config.get(
                                "wegscheider_cyclicity_enabled",
                                PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                            )
                        ),
                    )
                parameter_names = solver_context.owner_parameter_names_by_set_id.get(set_id_s)
                if parameter_names is None:
                    parameter_names = self._deps.slider_runtime_parameter_names(set_id=set_id_s)
                contained_owner_identity_by_set_id[set_id_s] = self._deps.preview_contained_owner_identity(
                    owner_mechanism_text=str(request_mechanism_text or mechanism_context.owner_full_dsl),
                    solver_config=solver_context.solver_config,
                    t_end=float(solver_context.t_end),
                    set_id=set_id_s,
                    parameter_names=parameter_names,
                    simulation_identity=simulation_identity_by_set_id.get(set_id_s),
                )
            else:
                mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                    simulation_identity=identity,
                )
                contained_owner_identity_by_set_id[set_id_s] = self._deps.ordinary_contained_owner_identity(
                    owner_mechanism_text=mechanism_context.owner_full_dsl,
                    solver_config=solver_context.solver_config,
                    t_end=float(solver_context.t_end),
                    set_id=set_id_s,
                    simulation_identity=simulation_identity_by_set_id.get(set_id_s),
                )

        scope_identity = SimulationScopeIdentity.build(
            queue_ids=mechanism_context.queue_ids,
            identity_by_set_id={set_id: payload for set_id, payload in simulation_identity_by_set_id.items()},
        )
        cache_key = self._ports.batch.batch_cache_key(scope_identity=scope_identity)
        if not bool(runtime_readiness_only):
            self._deps.record_run_cache_key(cache_key=cache_key, fast_mode=bool(fast_mode))

        for index, set_id in enumerate(mechanism_context.queue_ids):
            set_id_s = str(set_id)
            initials_dict = initials_by_set_id.get(set_id_s)
            if not isinstance(initials_dict, dict):
                continue
            set_name = (
                str(mechanism_context.queue_names[index])
                if index < len(mechanism_context.queue_names)
                else set_id_s
            )
            prepared_execution_payload = solver_context.execution_prepared_payload_by_set_id.get(set_id_s)
            dispatch_plan = build_batch_set_dispatch_plan(
                BatchSetDispatchInput(
                    set_id=set_id_s,
                    set_name=set_name,
                    fast_mode=bool(fast_mode),
                    t_end=float(solver_context.t_end),
                    solver_config=solver_context.solver_config,
                    cache_key=str(cache_key),
                    scope_identity=scope_identity.to_payload(),
                    queue_ids=[str(queue_id) for queue_id in mechanism_context.queue_ids],
                    initials=dict(initials_dict),
                    mechanism_text=mechanism_text_by_set_id.get(set_id_s, mechanism_context.owner_full_dsl),
                    simulation_identity=simulation_identity_by_set_id.get(set_id_s),
                    plan_payload=None,
                    preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(set_id_s, ""),
                    prepared_payload=(
                        dict(prepared_execution_payload)
                        if isinstance(prepared_execution_payload, Mapping)
                        else None
                    ),
                    parameter_overrides=(
                        parameter_overrides_by_set_id.get(set_id_s)
                        if bool(fast_mode)
                        else None
                    ),
                    intervention_schedule=intervention_schedule_by_set_id.get(set_id_s),
                    contained_owner_identity=contained_owner_identity_by_set_id.get(set_id_s),
                    algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                )
            )
            if dispatch_plan.simulation_identity:
                simulation_identity_by_set_id[set_id_s] = dict(dispatch_plan.simulation_identity)
            if isinstance(dispatch_plan.plan_payload, dict):
                simulation_plan_by_set_id[set_id_s] = dict(dispatch_plan.plan_payload)

        return RunDispatchContext(
            simulation_plan_by_set_id=simulation_plan_by_set_id,
            mechanism_text_by_set_id=mechanism_text_by_set_id,
            mechanism_signature_by_set_id=mechanism_signature_by_set_id,
            simulation_identity_by_set_id=simulation_identity_by_set_id,
            preview_batch_cache_token_by_set_id=preview_batch_cache_token_by_set_id,
            scope_identity=scope_identity,
            cache_key=str(cache_key),
        )


class SimulationRunPreparationOwner:
    def __init__(
        self,
        *,
        ports: SimulationRunPreparationPorts,
        dependencies: SimulationRunPreparationDependencies,
    ) -> None:
        self._mechanism_owner = SimulationRunMechanismPreparationOwner(
            ports=ports,
            dependencies=dependencies,
        )
        self._solver_owner = SimulationRunSolverPreparationOwner(
            ports=ports,
            dependencies=dependencies,
        )
        self._dispatch_owner = SimulationRunDispatchPreparationOwner(
            ports=ports,
            dependencies=dependencies,
        )

    def build_mechanism_context_or_abort(
        self,
        *,
        fast_mode: bool,
        request_id: int,
        batch_rows: Sequence[int],
        runtime_readiness_only: bool,
    ) -> RunMechanismContext | None:
        return self._mechanism_owner.build_mechanism_context_or_abort(
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            batch_rows=batch_rows,
            runtime_readiness_only=bool(runtime_readiness_only),
        )

    def build_solver_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
    ) -> RunSolverContext | None:
        return self._solver_owner.build_solver_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
        )

    def build_dispatch_context_or_abort(
        self,
        *,
        fast_mode: bool,
        runtime_readiness_only: bool,
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
    ) -> RunDispatchContext | None:
        return self._dispatch_owner.build_dispatch_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=bool(runtime_readiness_only),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
        )
