from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from kindred.gui.controllers.batch_run_context_owner import BatchCompletionCleanupState


@dataclass(frozen=True, slots=True)
class SimulationModalError:
    title: str
    message: str
    details: str | None = None


@dataclass(frozen=True, slots=True)
class SimulationLifecycleEffects:
    release_worker: bool = False
    cleanup_lane_pool: bool = False
    shutdown_lane_pool: bool = False
    lane_pool_force_terminate: bool = False
    keep_lane_pool_alive: bool = False
    clear_pending_plot_updates: bool = False
    stale_fast_handoff_after_display: bool = False
    close_contained_owner: bool = False
    close_contained_fast_mode: bool = False
    close_contained_kill: bool = False
    clear_shutdown_request: bool = False
    clear_pending_preview_plot_updates: bool = False
    reset_slider_triggered: bool = False
    simulation_running: bool | None = None
    slider_simulation_active: bool | None = None
    run_enabled: bool | None = None
    stop_enabled: bool | None = None
    status_text: str | None = None
    progress_value: int | None = None
    algebra_status_text: str | None = None
    clear_algebra_status: bool = False
    repaint_widgets: bool = False
    stop_debounce_timers: bool = False
    schedule_deferred_preview_replay: bool = False
    deferred_replay_stop_timers: bool = True
    apply_explicit_failure_pending_replay: bool = False
    invalidate_failed_pending_init_results: bool = False
    show_preview_unavailable_status: str | None = None
    modal_error: SimulationModalError | None = None


class SimulationLifecycleEffectOwner:
    """Owns completion/error lifecycle decisions as typed effects."""

    @staticmethod
    def progress_update(
        *,
        progress_value: int | None = None,
        status_text: str | None = None,
        repaint_widgets: bool = False,
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            progress_value=progress_value,
            status_text=status_text,
            repaint_widgets=bool(repaint_widgets),
        )

    @staticmethod
    def algebra_status_effect(
        *,
        species_names: Sequence[object],
        base_species_count: int | None,
        algebra_errors: object,
    ) -> SimulationLifecycleEffects:
        if not isinstance(algebra_errors, list) or not algebra_errors:
            return SimulationLifecycleEffects(algebra_status_text="")
        ok = max(0, len(list(species_names or ())) - int(base_species_count or 0))
        err = len([error for error in algebra_errors if isinstance(error, Mapping)])
        return SimulationLifecycleEffects(
            algebra_status_text=f"Algebra: {ok} ok, {err} error" + ("s" if err != 1 else "")
        )

    @staticmethod
    def completion_status_effect(
        *,
        species_count: int,
        point_count: int,
        failed_set_ids: Sequence[object],
        is_preview: bool,
    ) -> SimulationLifecycleEffects:
        failed_count = len(tuple(failed_set_ids or ()))
        if failed_count and not bool(is_preview):
            status_text = f"Batch completed with {failed_count} failed set(s)"
        else:
            status_text = f"Simulation complete: {int(species_count)} species, {int(point_count)} points"
        return SimulationLifecycleEffects(
            progress_value=100,
            status_text=status_text,
            repaint_widgets=True,
        )

    @staticmethod
    def superseded_fast_completion_effects(
        *,
        deactivate_context_immediately: bool,
        schedule_pending_preview_run: bool,
        reset_status_progress: bool,
        display_current_preview: bool,
        cleanup_state: BatchCompletionCleanupState,
    ) -> SimulationLifecycleEffects:
        if bool(schedule_pending_preview_run) and not bool(display_current_preview):
            return SimulationLifecycleEffects(
                release_worker=bool(deactivate_context_immediately),
                cleanup_lane_pool=bool(deactivate_context_immediately),
                keep_lane_pool_alive=bool(cleanup_state.parallel and cleanup_state.keep_lane_pool_alive),
                clear_pending_plot_updates=bool(deactivate_context_immediately),
                reset_slider_triggered=bool(deactivate_context_immediately),
                simulation_running=False if bool(deactivate_context_immediately) else None,
                slider_simulation_active=False if bool(deactivate_context_immediately) else None,
                run_enabled=True if bool(deactivate_context_immediately) else None,
                stop_enabled=False if bool(deactivate_context_immediately) else None,
                stop_debounce_timers=bool(deactivate_context_immediately),
                schedule_deferred_preview_replay=True,
                deferred_replay_stop_timers=False,
                clear_shutdown_request=True,
            )
        return SimulationLifecycleEffects(
            release_worker=bool(deactivate_context_immediately),
            cleanup_lane_pool=bool(deactivate_context_immediately),
            keep_lane_pool_alive=bool(cleanup_state.parallel and cleanup_state.keep_lane_pool_alive),
            clear_pending_plot_updates=bool(deactivate_context_immediately),
            reset_slider_triggered=bool(deactivate_context_immediately),
            simulation_running=False if bool(deactivate_context_immediately) else None,
            slider_simulation_active=False if bool(deactivate_context_immediately) else None,
            run_enabled=True if bool(deactivate_context_immediately) else None,
            stop_enabled=False if bool(deactivate_context_immediately) else None,
            progress_value=0 if bool(reset_status_progress) else None,
            status_text="Ready" if bool(reset_status_progress) else None,
            stop_debounce_timers=bool(deactivate_context_immediately),
            clear_shutdown_request=True,
        )

    @staticmethod
    def successful_completion_final_effects(
        *,
        cleanup_state: BatchCompletionCleanupState,
        stale_fast_handoff_after_display: bool,
        has_deferred_preview_replay: bool,
        shutdown_requested: bool,
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            release_worker=True,
            cleanup_lane_pool=True,
            keep_lane_pool_alive=bool(cleanup_state.parallel and cleanup_state.keep_lane_pool_alive),
            stale_fast_handoff_after_display=bool(stale_fast_handoff_after_display),
            reset_slider_triggered=True,
            simulation_running=False,
            slider_simulation_active=False,
            run_enabled=True,
            stop_enabled=False,
            schedule_deferred_preview_replay=bool(has_deferred_preview_replay and not shutdown_requested),
            clear_shutdown_request=True,
        )

    @staticmethod
    def serial_batch_continue_effects() -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(release_worker=True)

    @staticmethod
    def superseded_fast_error_effects(
        *,
        deactivate_context_immediately: bool,
        schedule_pending_preview_run: bool,
        reset_status_progress: bool,
    ) -> SimulationLifecycleEffects:
        if bool(schedule_pending_preview_run):
            return SimulationLifecycleEffects(
                release_worker=bool(deactivate_context_immediately),
                shutdown_lane_pool=bool(deactivate_context_immediately),
                lane_pool_force_terminate=False,
                clear_shutdown_request=bool(deactivate_context_immediately),
                reset_slider_triggered=bool(deactivate_context_immediately),
                simulation_running=False if bool(deactivate_context_immediately) else None,
                slider_simulation_active=False if bool(deactivate_context_immediately) else None,
                run_enabled=True if bool(deactivate_context_immediately) else None,
                stop_enabled=False if bool(deactivate_context_immediately) else None,
                stop_debounce_timers=bool(deactivate_context_immediately),
                schedule_deferred_preview_replay=True,
                deferred_replay_stop_timers=False,
            )
        return SimulationLifecycleEffects(
            release_worker=bool(deactivate_context_immediately),
            shutdown_lane_pool=bool(deactivate_context_immediately),
            lane_pool_force_terminate=False,
            clear_shutdown_request=bool(deactivate_context_immediately),
            reset_slider_triggered=bool(deactivate_context_immediately),
            simulation_running=False if bool(deactivate_context_immediately) else None,
            slider_simulation_active=False if bool(deactivate_context_immediately) else None,
            run_enabled=True if bool(deactivate_context_immediately) else None,
            stop_enabled=False if bool(deactivate_context_immediately) else None,
            progress_value=0 if bool(reset_status_progress) else None,
            status_text="Ready" if bool(reset_status_progress) else None,
            stop_debounce_timers=bool(deactivate_context_immediately),
        )

    @staticmethod
    def terminal_error_effects(
        *,
        cancelled: bool,
        error_text: str,
        error_detail_text: str,
        fast_mode: bool,
        has_deferred_preview_replay: bool,
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            release_worker=True,
            shutdown_lane_pool=True,
            lane_pool_force_terminate=True,
            close_contained_owner=True,
            close_contained_fast_mode=bool(fast_mode),
            close_contained_kill=True,
            clear_shutdown_request=True,
            status_text=("Simulation cancelled by user" if bool(cancelled) else "Simulation failed"),
            progress_value=0,
            algebra_status_text="",
            simulation_running=False,
            slider_simulation_active=False,
            run_enabled=True,
            stop_enabled=False,
            reset_slider_triggered=True,
            schedule_deferred_preview_replay=bool(cancelled and has_deferred_preview_replay),
            apply_explicit_failure_pending_replay=not bool(cancelled),
            invalidate_failed_pending_init_results=True,
            modal_error=(
                None
                if bool(cancelled)
                else SimulationModalError(
                    title="Simulation Error",
                    message=f"Simulation failed:\n\n{error_text}",
                    details=error_detail_text or None,
                )
            ),
        )

    @staticmethod
    def current_preview_failure_effects(*, status_text: str) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            release_worker=True,
            shutdown_lane_pool=True,
            lane_pool_force_terminate=True,
            close_contained_owner=True,
            close_contained_fast_mode=True,
            close_contained_kill=True,
            clear_shutdown_request=True,
            clear_pending_preview_plot_updates=True,
            reset_slider_triggered=True,
            simulation_running=False,
            slider_simulation_active=False,
            run_enabled=True,
            stop_enabled=False,
            progress_value=0,
            algebra_status_text="",
            show_preview_unavailable_status=str(status_text),
            status_text=str(status_text),
        )
