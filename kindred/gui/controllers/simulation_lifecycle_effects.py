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
    algebra_status_details: str | None = None
    clear_algebra_status: bool = False
    repaint_widgets: bool = False
    stop_debounce_timers: bool = False
    modal_error: SimulationModalError | None = None
    runtime_cancel_kind: str = ""
    runtime_display_kind: str = ""


class SimulationLifecycleEffectOwner:
    """Owns completion/error lifecycle decisions as typed effects."""

    @staticmethod
    def runtime_display_kind_for_cleanup(
        cleanup_state: BatchCompletionCleanupState,
        *,
        stale_fast_handoff_after_display: bool = False,
    ) -> str:
        if bool(stale_fast_handoff_after_display):
            return "stale_fast"
        if bool(
            (cleanup_state.runtime_task_queue or cleanup_state.parallel)
            and cleanup_state.keep_lane_pool_alive
        ):
            return "success"
        return "scoped_failure"

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
    def completion_without_result_ui_effects(
        *,
        summary: object,
        status_text: str,
    ) -> SimulationLifecycleEffects:
        failed_set_ids = tuple(getattr(summary, "failed_set_ids", ()) or ())
        fast_mode = bool(getattr(summary, "fast_mode", False))
        if failed_set_ids and not fast_mode:
            failed_count = len(failed_set_ids)
            return SimulationLifecycleEffects(
                progress_value=100,
                status_text=f"Batch completed with {failed_count} failed set(s)",
                repaint_widgets=True,
            )
        if bool(getattr(summary, "has_truthful_success", False)):
            return SimulationLifecycleEffects(
                progress_value=100,
                status_text=str(status_text),
                repaint_widgets=True,
            )
        return SimulationLifecycleEffects(
            progress_value=0,
            status_text="Ready",
            repaint_widgets=True,
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
        first = next((error for error in algebra_errors if isinstance(error, Mapping)), None)
        first_summary = SimulationLifecycleEffectOwner._format_algebra_error_summary(first)
        details = SimulationLifecycleEffectOwner._format_algebra_error_details(algebra_errors)
        summary = f"Algebra: {ok} ok, {err} error" + ("s" if err != 1 else "")
        if first_summary:
            summary = f"{summary} - {first_summary}"
        return SimulationLifecycleEffects(
            algebra_status_text=summary,
            algebra_status_details=details,
        )

    @staticmethod
    def _format_algebra_error_summary(error: Mapping[object, object] | None) -> str:
        if not isinstance(error, Mapping):
            return ""
        message = str(error.get("message") or "").strip()
        if not message:
            return ""
        name = str(error.get("name") or "").strip()
        if name:
            return f"{name}: {message}"
        return message

    @staticmethod
    def _format_algebra_error_details(algebra_errors: Sequence[object]) -> str:
        lines: list[str] = []
        for index, error in enumerate(algebra_errors or (), start=1):
            if not isinstance(error, Mapping):
                continue
            summary = SimulationLifecycleEffectOwner._format_algebra_error_summary(error)
            if not summary:
                continue
            location_parts = []
            if error.get("line") is not None:
                location_parts.append(f"line {error.get('line')}")
            if error.get("col") is not None:
                location_parts.append(f"col {error.get('col')}")
            location = f" ({', '.join(location_parts)})" if location_parts else ""
            lines.append(f"{index}. {summary}{location}")
            line_text = str(error.get("line_text") or "").strip()
            if line_text:
                lines.append(f"   {line_text}")
        return "\n".join(lines)

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
        reset_status_progress: bool,
        display_current_preview: bool,
        cleanup_state: BatchCompletionCleanupState,
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            cleanup_lane_pool=bool(deactivate_context_immediately),
            runtime_display_kind=SimulationLifecycleEffectOwner.runtime_display_kind_for_cleanup(cleanup_state),
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
        shutdown_requested: bool,
    ) -> SimulationLifecycleEffects:
        _ = shutdown_requested
        return SimulationLifecycleEffects(
            cleanup_lane_pool=True,
            runtime_display_kind=SimulationLifecycleEffectOwner.runtime_display_kind_for_cleanup(
                cleanup_state,
                stale_fast_handoff_after_display=bool(stale_fast_handoff_after_display),
            ),
            reset_slider_triggered=True,
            simulation_running=False,
            slider_simulation_active=False,
            run_enabled=True,
            stop_enabled=False,
            clear_shutdown_request=True,
        )

    @staticmethod
    def superseded_fast_error_effects(
        *,
        deactivate_context_immediately: bool,
        reset_status_progress: bool,
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            shutdown_lane_pool=bool(deactivate_context_immediately),
            runtime_cancel_kind="soft_shutdown",
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
    ) -> SimulationLifecycleEffects:
        return SimulationLifecycleEffects(
            shutdown_lane_pool=True,
            runtime_cancel_kind="stop" if bool(cancelled) else "terminal_failure",
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
            shutdown_lane_pool=True,
            runtime_cancel_kind="preview_failure",
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
            status_text=str(status_text),
        )
