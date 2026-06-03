from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from kindred.core.batch_parallel import batch_mechanism_signature, compute_effective_batch_workers
from kindred.core.intervention_schedule import intervention_schedule_identity_fingerprints
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulation_identity import (
    SimulationScopeIdentity,
    canonical_initials_fingerprint,
    coerce_simulation_identity,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy
from kindred.gui.controllers.batch_run_context_owner import BatchRunStartRequest
from kindred.gui.controllers.batch_dispatch_plan import (
    BatchSetDispatchInput,
    build_batch_set_dispatch_plan,
    execution_request_payload_from_plan,
)
from kindred.gui.controllers.preview_target_identity import normalize_preview_target_set_ids
from kindred.gui.controllers.runtime_lane_allocation import (
    PreparedRuntimeRequestSet,
    RuntimeCompatibilityKey,
    RuntimeLaunchIntent,
    RuntimePreparationBlockedReason,
    RuntimeTaskDescriptor,
)
from kindred.gui.ports import CompletedRunDisplayIntent
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
    has_runtime_parameter_values: bool
    primary: object | None
    primary_set_id: str | None
    base_source: MechanismAuthoringSource
    owner_full_dsl: str
    full_dsl: str
    preview_owner_epoch: int | None


@dataclass
class RunSolverContext:
    solver_config: Dict[str, Any]
    t_end: float
    prepared_payload: Dict[str, Any] | None
    prepared_payload_by_set_id: Dict[str, Dict[str, Any]]
    execution_prepared_payload_by_set_id: Dict[str, Dict[str, Any]]
    runtime_parameter_names_by_set_id: Dict[str, List[str]]


@dataclass
class RunDispatchContext:
    simulation_plan_by_set_id: Dict[str, Dict[str, Any]]
    mechanism_text_by_set_id: Dict[str, str]
    mechanism_signature_by_set_id: Dict[str, str]
    simulation_identity_by_set_id: Dict[str, Dict[str, Any]]
    owned_species_by_set_id: Dict[str, tuple[str, ...]]
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
    clear_failed_fast_preview_ownership: Callable[[], None]
    clear_slider_triggered_preflight_state: Callable[..., None]
    requeue_preserved_pending_slider_replay_after_preflight_abort: Callable[[], None]
    record_nonfatal_exception: Callable[[str, BaseException], None]
    set_simulation_running: Callable[[bool], None]
    set_slider_simulation_active: Callable[[bool], None]
    runtime_parameter_names_for_set: Callable[..., Sequence[str]]
    pending_initials_for_run_source_set: Callable[..., Dict[str, Any]]
    simulation_identity_for_set: Callable[..., Any]
    resolved_initials_for_batch_row: Callable[..., Dict[str, Any]]
    runtime_parameter_values_for_set: Callable[..., Dict[str, Any]]
    preview_contained_owner_identity: Callable[..., Dict[str, Any]]
    ordinary_contained_owner_identity: Callable[..., Dict[str, Any]]
    record_run_cache_key: Callable[..., None]
    runtime_environment_key: Callable[[], str] = lambda: "contained-child-blas-limited"
    runtime_lane_budget: Callable[[], int] = lambda: int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
    max_parallel_batch_workers: Callable[[], int] = lambda: int(PROJECT_DEFAULTS["max_parallel_batch_workers"])


def build_fast_preview_solver_grid_context(
    *,
    initial_solver_name: Optional[str],
    num_points: int,
    fast_mode: bool,
    slider_points_override: Optional[int],
    slider_solver_override: Optional[str],
    slider_drag_active: bool,
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
    requested_show_set_ids: Sequence[str],
    requested_show_labels_by_set_id: Mapping[str, str] | None = None,
) -> RunStartContext:
    queue_ids = list(mechanism_context.queue_ids)
    requested_show_intent_set_ids = _deduped_nonempty_set_ids(requested_show_set_ids)
    requested_show_intent_labels_by_set_id = _completed_run_display_labels_by_set_id(
        requested_show_set_ids=requested_show_intent_set_ids,
        requested_show_labels_by_set_id=requested_show_labels_by_set_id,
        queue_ids=queue_ids,
        queue_names=mechanism_context.queue_names,
    )
    display_primary_set_id = _completed_run_display_primary_set_id(
        requested_show_set_ids=requested_show_intent_set_ids,
        preferred_set_id=mechanism_context.primary_set_id,
    )
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
        explicit_cache_preview_token=None,
        explicit_cache_preview_scope_set_ids=run_start_cache_decision.explicit_preview_scope_set_ids,
        explicit_cache_valid_set_ids=run_start_cache_decision.explicit_cache_valid_set_ids,
        explicit_cache_invalidated_set_ids=run_start_cache_decision.explicit_cache_invalidated_set_ids,
        preview_scope_set_ids=run_start_cache_decision.preview_scope_set_ids,
        preview_owner_epoch=mechanism_context.preview_owner_epoch,
        preview_batch_cache_token_by_set_id=dispatch_context.preview_batch_cache_token_by_set_id,
        computed_owned_species_by_set_id=dispatch_context.owned_species_by_set_id,
        completed_run_display_intent=CompletedRunDisplayIntent(
            requested_show_set_ids=requested_show_intent_set_ids,
            labels_by_set_id=requested_show_intent_labels_by_set_id,
            primary_set_id=display_primary_set_id,
            cache_key=str(dispatch_context.cache_key or ""),
            run_id=run_id,
            request_id=int(request_id),
            owned_species_by_set_id=_completed_run_display_owned_species_by_set_id(
                dispatch_context=dispatch_context,
                set_ids=requested_show_intent_set_ids,
            ),
            run_target_set_ids=tuple(str(set_id) for set_id in queue_ids if str(set_id)),
        ),
    )
    return RunStartContext(
        request=request,
        parallel_mode=bool(parallel_mode),
        effective_workers=int(effective_workers),
        run_id=run_id,
        run_sequence_id=int(run_sequence_id),
    )


def _deduped_nonempty_set_ids(set_ids: Sequence[str]) -> tuple[str, ...]:
    return normalize_preview_target_set_ids(set_ids)


def _completed_run_display_labels_by_set_id(
    *,
    requested_show_set_ids: Sequence[str],
    requested_show_labels_by_set_id: Mapping[str, str] | None,
    queue_ids: Sequence[str],
    queue_names: Sequence[str],
) -> Dict[str, str]:
    requested_labels = {
        str(set_id): str(label)
        for set_id, label in dict(requested_show_labels_by_set_id or {}).items()
        if str(set_id)
    }
    queue_labels = {
        str(set_id): (str(queue_names[index]) if index < len(queue_names) and str(queue_names[index]) else str(set_id))
        for index, set_id in enumerate(queue_ids)
        if str(set_id)
    }
    return {
        str(set_id): str(requested_labels.get(str(set_id)) or queue_labels.get(str(set_id)) or str(set_id))
        for set_id in requested_show_set_ids
        if str(set_id)
    }


def _completed_run_display_primary_set_id(
    *,
    requested_show_set_ids: Sequence[str],
    preferred_set_id: str | None,
) -> str:
    requested_show_ids = tuple(str(set_id) for set_id in requested_show_set_ids if str(set_id))
    preferred = str(preferred_set_id or "").strip()
    if preferred and preferred in set(requested_show_ids):
        return preferred
    return str(requested_show_ids[0]) if requested_show_ids else ""


def _completed_run_display_owned_species_by_set_id(
    *,
    dispatch_context: RunDispatchContext,
    set_ids: Sequence[str],
) -> Dict[str, tuple[str, ...]]:
    owned_by_set: Dict[str, tuple[str, ...]] = {}
    for set_id in set_ids:
        sid = str(set_id or "").strip()
        if not sid:
            continue
        owned_species = tuple(
            str(name)
            for name in dispatch_context.owned_species_by_set_id.get(sid, ())
            if str(name)
        )
        if owned_species:
            owned_by_set[sid] = owned_species
    return owned_by_set


def _owned_species_from_launch_mechanism_text(
    mechanism_text: str,
    *,
    temperature_K: float | None,
) -> tuple[str, ...]:
    if temperature_K is None:
        return ()
    try:
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism
        from kindred.core.units import UnitsModel

        mechanism = parse_dsl_to_mechanism(
            str(mechanism_text or ""),
            initials={},
            units=UnitsModel(temperature_K=float(temperature_K), energy_unit="kJ/mol"),
        )
        return tuple(str(name) for name in mechanism.species_names() if str(name))
    except Exception:
        return ()


def _semantic_temperature_from_solver_config(solver_config: Mapping[str, Any]) -> float | None:
    raw_temperature = solver_config.get("temperature_K")
    if raw_temperature is None:
        return None
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(temperature) or temperature <= 0.0:
        return None
    return temperature


def _owned_species_from_prepared_launch_payload(
    *prepared_payloads: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    for prepared_payload in prepared_payloads:
        if not isinstance(prepared_payload, Mapping):
            continue
        species_source = prepared_payload.get("species_names")
        if species_source is None:
            mechanism = prepared_payload.get("mechanism")
            species_names = getattr(mechanism, "species_names", None)
            if callable(species_names):
                try:
                    species_source = species_names()
                except Exception:
                    species_source = None
        try:
            owned_species = tuple(str(name) for name in (species_source or ()) if str(name))
        except Exception:
            owned_species = ()
        if owned_species:
            return owned_species
    return ()


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
        any_runtime_parameter_workspace = bool(self._ports.mechanism.has_local_runtime_parameter_values())
        has_runtime_parameter_values = bool(fast_mode) and bool(any_runtime_parameter_workspace)
        primary = self._ports.batch.batch_preferred_primary_set_id(batch_rows)
        primary_set_id = str(primary) if primary is not None else None

        try:
            selected_source = self._ports.mechanism.mechanism_source_for_run(fast_mode=bool(fast_mode))
        except RuntimeError as exc:
            if bool(runtime_readiness_only):
                return None
            status = "Preview mechanism has errors." if bool(fast_mode) else "Cannot run: mechanism has errors."
            self._ports.run_ui.set_status_text(status)
            if bool(fast_mode):
                self._deps.clear_failed_fast_preview_ownership()
            self._deps.set_simulation_running(False)
            self._ports.run_ui.set_run_button_enabled(True)
            self._ports.run_ui.set_stop_button_enabled(False)
            self._deps.set_slider_simulation_active(False)
            self._deps.clear_slider_triggered_preflight_state(fast_mode=bool(fast_mode))
            if not bool(fast_mode):
                self._deps.requeue_preserved_pending_slider_replay_after_preflight_abort()
            self._deps.record_nonfatal_exception("Mechanism source was not ready for run preparation", exc)
            return None
        rows = list(batch_rows)
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

        base_source = selected_source
        full_source = self._ports.mechanism.mechanism_source_for_run_set(
            base_source,
            set_id=primary_set_id,
            apply_parameter_overrides=bool(has_runtime_parameter_values),
            strip_initial_concentrations=True,
        )
        full_dsl = full_source.full_dsl
        owner_full_dsl = base_source.without_reaction_initial_concentrations().full_dsl

        if not full_dsl.strip():
            if bool(runtime_readiness_only):
                return None
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
            has_runtime_parameter_values=bool(has_runtime_parameter_values),
            primary=primary,
            primary_set_id=primary_set_id,
            base_source=base_source,
            owner_full_dsl=owner_full_dsl,
            full_dsl=full_dsl,
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
        runtime_parameter_names_by_set_id: Dict[str, List[str]] = {}
        if bool(fast_mode):
            target_runtime_set_ids = list(mechanism_context.queue_ids)
            if (not target_runtime_set_ids) and mechanism_context.primary_set_id:
                target_runtime_set_ids = [str(mechanism_context.primary_set_id)]
            for set_id in target_runtime_set_ids:
                runtime_parameter_names = self._deps.runtime_parameter_names_for_set(set_id=str(set_id))
                runtime_parameter_names_by_set_id[str(set_id)] = list(runtime_parameter_names)
            if mechanism_context.primary_set_id:
                prepared_payload = prepared_payload_by_set_id.get(str(mechanism_context.primary_set_id))
            if prepared_payload is None and prepared_payload_by_set_id:
                prepared_payload = dict(next(iter(prepared_payload_by_set_id.values())))

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
            runtime_parameter_names_by_set_id=runtime_parameter_names_by_set_id,
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
        owned_species_by_set_id: Dict[str, tuple[str, ...]] = {}
        solver_temperature_K = _semantic_temperature_from_solver_config(solver_context.solver_config)

        def _abort_invalid_intervention_schedule(set_id_s: str, exc: BaseException) -> None:
            if bool(runtime_readiness_only):
                raise ValueError(f"Invalid intervention schedule for set {set_id_s!r}: {exc}") from exc
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
                    (
                        intervention_schedule_declarative_fingerprint,
                        intervention_schedule_executable_fingerprint,
                    ) = intervention_schedule_identity_fingerprints(intervention_schedule)
                    identity = self._deps.simulation_identity_for_set(
                        set_id=set_id_s,
                        solver_config=solver_context.solver_config,
                        t_end=float(solver_context.t_end),
                        preview_batch_cache_token=preview_batch_cache_token_by_set_id.get(set_id_s, ""),
                        intervention_schedule_declarative_fingerprint=(
                            intervention_schedule_declarative_fingerprint
                        ),
                        intervention_schedule_executable_fingerprint=(
                            intervention_schedule_executable_fingerprint
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
                prepared_execution_payload = solver_context.execution_prepared_payload_by_set_id.get(set_id_s)
                prepared_payload = solver_context.prepared_payload_by_set_id.get(set_id_s)
                owned_species = _owned_species_from_prepared_launch_payload(
                    prepared_execution_payload,
                    prepared_payload,
                ) or _owned_species_from_launch_mechanism_text(
                    str(mechanism_context.owner_full_dsl or ""),
                    temperature_K=solver_temperature_K,
                )
                if owned_species:
                    owned_species_by_set_id[set_id_s] = owned_species
                continue
            row = int(mechanism_context.batch_rows[index])
            set_name = (
                str(mechanism_context.queue_names[index])
                if index < len(mechanism_context.queue_names)
                else str(set_id)
            )
            request_source = self._ports.mechanism.mechanism_source_for_run_set(
                mechanism_context.base_source,
                set_id=set_id_s,
                apply_parameter_overrides=bool(mechanism_context.has_runtime_parameter_values),
                strip_initial_concentrations=True,
            )
            request_mechanism_text = request_source.full_dsl
            mechanism_text_by_set_id[set_id_s] = str(request_mechanism_text)
            prepared_execution_payload = solver_context.execution_prepared_payload_by_set_id.get(set_id_s)
            prepared_payload = solver_context.prepared_payload_by_set_id.get(set_id_s)
            owned_species = _owned_species_from_prepared_launch_payload(
                prepared_execution_payload,
                prepared_payload,
            ) or _owned_species_from_launch_mechanism_text(
                str(request_mechanism_text),
                temperature_K=solver_temperature_K,
            )
            if owned_species:
                owned_species_by_set_id[set_id_s] = owned_species
            try:
                intervention_schedule = _submitted_intervention_schedule_from_text(str(request_mechanism_text))
                (
                    intervention_schedule_declarative_fingerprint,
                    intervention_schedule_executable_fingerprint,
                ) = intervention_schedule_identity_fingerprints(intervention_schedule)
            except Exception as exc:
                _abort_invalid_intervention_schedule(set_id_s, exc)
                return None
            if intervention_schedule is not None:
                intervention_schedule_by_set_id[set_id_s] = intervention_schedule.to_payload()
            try:
                pending_initials = {}
                if bool(fast_mode):
                    pending_initials = dict(
                        self._deps.pending_initials_for_run_source_set(
                            mechanism_context.base_source,
                            set_name=set_name,
                        )
                    )
                initials_dict = self._deps.resolved_initials_for_batch_row(
                    row=row,
                    include_preview_initials=bool(fast_mode),
                    pending_initials=pending_initials,
                )
            except Exception as exc:
                if bool(runtime_readiness_only):
                    return None
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
                    intervention_schedule_declarative_fingerprint=(
                        intervention_schedule_declarative_fingerprint
                    ),
                    intervention_schedule_executable_fingerprint=(
                        intervention_schedule_executable_fingerprint
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
                parameter_overrides_by_set_id[set_id_s] = self._deps.runtime_parameter_values_for_set(
                    set_id=set_id_s
                )
                if isinstance(prepared_execution_payload, dict):
                    mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                        simulation_identity=identity,
                    )
                elif solver_temperature_K is not None:
                    mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                        mechanism_text=str(request_mechanism_text),
                        temperature_K=solver_temperature_K,
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
                else:
                    mechanism_signature_by_set_id[set_id_s] = batch_mechanism_signature(
                        simulation_identity=identity,
                    )
                contained_owner_identity_by_set_id[set_id_s] = self._deps.preview_contained_owner_identity(
                    owner_mechanism_text=str(request_mechanism_text or mechanism_context.owner_full_dsl),
                    solver_config=solver_context.solver_config,
                    t_end=float(solver_context.t_end),
                    set_id=set_id_s,
                    parameter_names=solver_context.runtime_parameter_names_by_set_id.get(set_id_s, ()),
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
            owned_species_by_set_id=owned_species_by_set_id,
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
        self._ports = ports
        self._deps = dependencies
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

    def prepare_runtime_request_set(
        self,
        *,
        intent: RuntimeLaunchIntent,
        fast_mode: bool,
    ) -> PreparedRuntimeRequestSet:
        rows_or_block = self._runtime_request_rows(intent.rows)
        if isinstance(rows_or_block, RuntimePreparationBlockedReason):
            return self._blocked_prepared_request_set(
                intent=intent,
                blocked_reason=rows_or_block,
            )
        rows = rows_or_block
        mechanism_context = self.build_mechanism_context_or_abort(
            fast_mode=bool(fast_mode),
            request_id=int(intent.request_token or 0),
            batch_rows=rows,
            runtime_readiness_only=True,
        )
        if mechanism_context is None:
            return self._blocked_prepared_request_set(
                intent=intent,
                blocked_reason=RuntimePreparationBlockedReason(
                    source="mechanism",
                    code="mechanism_preparation_blocked",
                    message="Simulation mechanism is not ready for runtime preparation.",
                    rows=tuple(rows),
                ),
            )
        solver_context = self.build_solver_context_or_abort(
            fast_mode=bool(fast_mode),
            runtime_readiness_only=True,
            mechanism_context=mechanism_context,
        )
        if solver_context is None:
            return self._blocked_prepared_request_set(
                intent=intent,
                blocked_reason=RuntimePreparationBlockedReason(
                    source="solver",
                    code="solver_preparation_blocked",
                    message="Simulation solver settings are not ready for runtime preparation.",
                    rows=tuple(rows),
                    set_ids=tuple(mechanism_context.queue_ids),
                ),
            )
        try:
            dispatch_context = self.build_dispatch_context_or_abort(
                fast_mode=bool(fast_mode),
                runtime_readiness_only=True,
                mechanism_context=mechanism_context,
                solver_context=solver_context,
            )
        except Exception as exc:
            return self._blocked_prepared_request_set(
                intent=intent,
                blocked_reason=RuntimePreparationBlockedReason(
                    source="dispatch",
                    code="dispatch_preparation_error",
                    message=f"Simulation dispatch preparation failed: {exc}",
                    rows=tuple(rows),
                    set_ids=tuple(mechanism_context.queue_ids),
                    retryable=True,
                ),
            )
        if dispatch_context is None:
            return self._blocked_prepared_request_set(
                intent=intent,
                blocked_reason=RuntimePreparationBlockedReason(
                    source="dispatch",
                    code="dispatch_preparation_blocked",
                    message="Simulation dispatch is not ready for runtime preparation.",
                    rows=tuple(rows),
                    set_ids=tuple(mechanism_context.queue_ids),
                ),
            )
        compatibility_key = self._runtime_compatibility_key(
            fast_mode=bool(fast_mode),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
            dispatch_context=dispatch_context,
        )
        descriptors = self._runtime_task_descriptors(
            intent=intent,
            compatibility_key=compatibility_key,
            rows=tuple(rows),
            mechanism_context=mechanism_context,
            solver_context=solver_context,
            dispatch_context=dispatch_context,
        )
        if not descriptors:
            return self._blocked_prepared_request_set(
                intent=intent,
                compatibility_key=compatibility_key,
                blocked_reason=RuntimePreparationBlockedReason(
                    source="dispatch",
                    code="no_task_descriptors",
                    message="Simulation dispatch did not produce runtime task descriptors.",
                    rows=tuple(rows),
                    set_ids=tuple(mechanism_context.queue_ids),
                ),
            )
        preferred = self._runtime_preferred_lane_capacity(rows=rows, descriptor_count=len(descriptors))
        required = preferred if bool(fast_mode) else 1
        return PreparedRuntimeRequestSet(
            intent=intent,
            compatibility_key=compatibility_key,
            task_descriptors=descriptors,
            required_lane_capacity=required,
            preferred_lane_capacity=preferred,
        )

    def _runtime_preferred_lane_capacity(
        self,
        *,
        rows: Sequence[int],
        descriptor_count: int,
    ) -> int:
        try:
            lane_budget = int(self._deps.runtime_lane_budget())
        except Exception:
            lane_budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
        try:
            max_workers = int(self._deps.max_parallel_batch_workers())
        except Exception:
            max_workers = int(PROJECT_DEFAULTS["max_parallel_batch_workers"])
        row_count = max(0, len(tuple(rows or ())))
        descriptor_limit = max(1, int(descriptor_count or 1))
        return max(
            1,
            min(
                max(1, int(lane_budget)),
                descriptor_limit,
                int(
                    compute_effective_batch_workers(
                        num_sets=row_count,
                        max_parallel_workers=max(1, int(max_workers)),
                    )
                ),
            ),
        )

    def _runtime_request_rows(
        self,
        batch_rows: Sequence[int],
    ) -> list[int] | RuntimePreparationBlockedReason:
        try:
            row_count = int(self._ports.batch.batch_store_row_count())
        except Exception as exc:
            self._deps.record_nonfatal_exception(
                "Failed to inspect rows for runtime request preparation",
                exc,
            )
            return RuntimePreparationBlockedReason(
                source="batch",
                code="row_count_unavailable",
                message="Simulation rows are not available for runtime preparation.",
                retryable=True,
            )
        rows: list[int] = []
        invalid_rows: list[int] = []
        seen: set[int] = set()
        for row in batch_rows or ():
            try:
                row_i = int(row)
            except (TypeError, ValueError, OverflowError):
                continue
            if row_i in seen:
                continue
            seen.add(row_i)
            if 0 <= row_i < int(row_count):
                rows.append(row_i)
            else:
                invalid_rows.append(row_i)
        if invalid_rows:
            return RuntimePreparationBlockedReason(
                source="batch",
                code="invalid_rows",
                message="Selected simulation rows are no longer available.",
                rows=tuple(invalid_rows),
                retryable=True,
            )
        if not rows:
            return RuntimePreparationBlockedReason(
                source="batch",
                code="no_rows",
                message="No simulation rows are selected for runtime preparation.",
                retryable=False,
            )
        try:
            invalid = self._ports.batch.batch_model_validate_rows(rows)
        except Exception as exc:
            self._deps.record_nonfatal_exception(
                "Failed to validate rows for runtime request preparation",
                exc,
            )
            return RuntimePreparationBlockedReason(
                source="batch",
                code="row_validation_failed",
                message=f"Simulation rows could not be validated. {exc}",
                rows=tuple(rows),
                retryable=True,
            )
        if invalid:
            return RuntimePreparationBlockedReason(
                source="batch",
                code="invalid_initial_conditions",
                message="Selected simulation rows have invalid initial conditions.",
                rows=tuple(rows),
                retryable=True,
            )
        return rows

    def _runtime_compatibility_key(
        self,
        *,
        fast_mode: bool,
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
        dispatch_context: RunDispatchContext,
    ) -> RuntimeCompatibilityKey:
        runtime_parameter_names = tuple(
            sorted(
                {
                    str(name)
                    for names in dict(solver_context.runtime_parameter_names_by_set_id or {}).values()
                    for name in list(names or ())
                    if str(name)
                }
            )
        )
        structural_payload = {
            "execution_profile": "preview" if bool(fast_mode) else "explicit",
            "runtime_parameter_names": list(runtime_parameter_names),
            "solver_config": dict(solver_context.solver_config or {}),
            "t_end": float(solver_context.t_end),
            "schema_key": "simulation-plan-v1",
        }
        structural_digest = hashlib.sha256(
            json.dumps(structural_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        environment_key = "contained-child-blas-limited"
        deps = getattr(self, "_deps", None)
        environment_key_getter = getattr(deps, "runtime_environment_key", None)
        if callable(environment_key_getter):
            try:
                environment_key = str(environment_key_getter() or environment_key)
            except Exception:
                environment_key = "runtime-environment-unavailable"
        return RuntimeCompatibilityKey(
            structural_digest=structural_digest,
            runtime_parameter_names=runtime_parameter_names,
            execution_profile="preview" if bool(fast_mode) else "explicit",
            environment_key=environment_key,
            schema_key="simulation-plan-v1",
        )

    def _runtime_task_descriptors(
        self,
        *,
        intent: RuntimeLaunchIntent,
        compatibility_key: RuntimeCompatibilityKey,
        rows: Sequence[int],
        mechanism_context: RunMechanismContext,
        solver_context: RunSolverContext,
        dispatch_context: RunDispatchContext,
    ) -> tuple[RuntimeTaskDescriptor, ...]:
        descriptors: list[RuntimeTaskDescriptor] = []
        runtime_epochs = dict(intent.runtime_input_epochs or {})
        set_ids_by_row = {
            int(row): str(set_id)
            for row, set_id in zip(rows, mechanism_context.queue_ids)
            if str(set_id)
        }
        set_labels_by_row = {
            int(row): str(set_name)
            for row, set_name in zip(rows, mechanism_context.queue_names)
            if str(set_name)
        }
        for row in rows:
            set_id = str(set_ids_by_row.get(int(row)) or "")
            if not set_id:
                continue
            plan_payload = dispatch_context.simulation_plan_by_set_id.get(set_id)
            if not isinstance(plan_payload, Mapping):
                continue
            parameter_overrides = {}
            prepared_payload = solver_context.execution_prepared_payload_by_set_id.get(set_id)
            if isinstance(prepared_payload, Mapping):
                parameter_overrides = dict(prepared_payload.get("parameter_overrides") or {})
            if not parameter_overrides and isinstance(plan_payload, Mapping):
                try:
                    plan_execution_request = execution_request_payload_from_plan(plan_payload)
                except Exception:
                    plan_execution_request = None
                if isinstance(plan_execution_request, Mapping):
                    parameter_overrides = dict(
                        plan_execution_request.get("parameter_overrides") or {}
                    )
            owned_species = tuple(
                str(name)
                for name in dispatch_context.owned_species_by_set_id.get(set_id, ())
                if str(name)
            )
            preview_batch_cache_token = str(
                dispatch_context.preview_batch_cache_token_by_set_id.get(set_id) or ""
            )
            exact_payload = {
                "row": int(row),
                "set_id": set_id,
                "request_token": intent.request_token,
                "preview_request_id": intent.preview_request_id,
                "preview_epoch": intent.preview_epoch,
                "cache_key": str(dispatch_context.cache_key),
                "mechanism_text": str(dispatch_context.mechanism_text_by_set_id.get(set_id) or ""),
                "mechanism_signature": str(dispatch_context.mechanism_signature_by_set_id.get(set_id) or ""),
                "simulation_identity": dict(dispatch_context.simulation_identity_by_set_id.get(set_id) or {}),
                "owned_species": list(owned_species),
                "preview_batch_cache_token": preview_batch_cache_token,
                "plan_payload": dict(plan_payload),
                "parameter_overrides": parameter_overrides,
                "runtime_input_epochs": runtime_epochs,
            }
            exact_hash = hashlib.sha256(
                json.dumps(exact_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
            descriptors.append(
                RuntimeTaskDescriptor(
                    task_id=f"{intent.intent_kind}:{set_id}:{int(row)}",
                    row=int(row),
                    set_id=set_id,
                    request_token=int(intent.request_token) if intent.request_token is not None else None,
                    compatibility_key=compatibility_key,
                    exact_descriptor_hash=exact_hash,
                    plan_payload=dict(plan_payload),
                    parameter_overrides=parameter_overrides,
                    preview_request_id=int(intent.preview_request_id)
                    if intent.preview_request_id is not None
                    else None,
                    preview_epoch=int(intent.preview_epoch)
                    if intent.preview_epoch is not None
                    else None,
                    runtime_input_epochs=runtime_epochs,
                    cache_key=str(dispatch_context.cache_key),
                    set_label=str(set_labels_by_row.get(int(row)) or set_id),
                    mechanism_text=str(dispatch_context.mechanism_text_by_set_id.get(set_id) or ""),
                    mechanism_signature=str(dispatch_context.mechanism_signature_by_set_id.get(set_id) or ""),
                    simulation_identity=dict(dispatch_context.simulation_identity_by_set_id.get(set_id) or {}),
                    owned_species=owned_species,
                    preview_batch_cache_token=preview_batch_cache_token,
                )
            )
        return tuple(descriptors)

    @staticmethod
    def _blocked_prepared_request_set(
        *,
        intent: RuntimeLaunchIntent,
        blocked_reason: RuntimePreparationBlockedReason,
        compatibility_key: RuntimeCompatibilityKey | None = None,
    ) -> PreparedRuntimeRequestSet:
        key = compatibility_key or RuntimeCompatibilityKey(
            structural_digest="blocked",
            execution_profile=str(intent.intent_kind or "ordinary"),
        )
        return PreparedRuntimeRequestSet(
            intent=intent,
            compatibility_key=key,
            task_descriptors=(),
            required_lane_capacity=1,
            preferred_lane_capacity=1,
            blocked_reason=blocked_reason,
        )
