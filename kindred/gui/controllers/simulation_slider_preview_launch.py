from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, replace
import logging
from typing import Any, Callable

from PySide6 import QtCore

from kindred.gui.controllers.simulation_run_state import PendingSliderPreviewLaunchState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliderPreviewLaunchRunRequest:
    request_id: int
    batch_rows: tuple[int, ...]
    reuse_parallel_lane_pool: bool = True


@dataclass(frozen=True)
class SimulationSliderPreviewLaunchDependencies:
    set_discarded_preview_generation: Callable[[int | None], None]
    worker_is_running: Callable[[Any], bool]
    clear_pending_slider_preview_replay: Callable[..., None]
    next_slider_preview_request_id: Callable[[], int]
    queue_pending_slider_preview_replay: Callable[..., None]
    has_active_explicit_simulation: Callable[[], bool]
    has_active_parallel_batch_work: Callable[[], bool]
    supersede_parallel_batch_run_soft: Callable[[], object]
    prune_stopped_owned_simulation_workers: Callable[[], None]
    has_running_owned_simulation_workers: Callable[[], bool]
    slider_target_rows_for_dispatch: Callable[..., list[int]]
    slider_preview_uses_parallel_batch_runtime: Callable[..., bool]
    slider_preview_runtime_snapshot: Callable[..., Any]
    ensure_parallel_batch_pool_eagerly_created: Callable[..., None]
    ensure_interactive_simulation_runtime_available_for_mode: Callable[..., None]
    mark_request_started: Callable[[int], int]
    retry_slider_preview_launch: Callable[[], None]


class SimulationSliderPreviewLaunchOwner:
    def __init__(
        self,
        *,
        ui: Any,
        run_state: Any,
        batch_context_owner: Any,
        dependencies: SimulationSliderPreviewLaunchDependencies,
    ) -> None:
        self._ui = ui
        self._run_state = run_state
        self._batch_context_owner = batch_context_owner
        self._deps = dependencies

    def _pending_launch(self) -> PendingSliderPreviewLaunchState:
        replay = getattr(self._run_state, "pending_slider_preview_launch", None)
        if isinstance(replay, PendingSliderPreviewLaunchState):
            return replay
        normalized = PendingSliderPreviewLaunchState()
        self._run_state.pending_slider_preview_launch = normalized
        return normalized

    @staticmethod
    def _has_launch_state(replay: PendingSliderPreviewLaunchState) -> bool:
        return bool(replay.active or replay.request_id is not None or replay.target_set_ids)

    def _queue_current_replay(
        self,
        *,
        request_id: int,
        pending_target_set_ids: list[str],
    ) -> None:
        target_set_ids = [str(set_id) for set_id in pending_target_set_ids if str(set_id)]
        if not target_set_ids:
            try:
                target_set_ids = [
                    str(set_id)
                    for set_id in (self._ui.batch.batch_set_ids_for_scope("selected") or ())
                    if str(set_id)
                ]
            except Exception:
                target_set_ids = []
        self._deps.queue_pending_slider_preview_replay(
            target_set_ids=target_set_ids,
            request_id=int(request_id),
        )

    def run_from_slider(self) -> SliderPreviewLaunchRunRequest | None:
        replay = self._pending_launch()
        if not self._has_launch_state(replay):
            if replay.handoff_queued:
                self._run_state.pending_slider_preview_launch = replace(replay, handoff_queued=False)
            return
        if replay.handoff_queued:
            replay = replace(replay, active=True, handoff_queued=False)
            self._run_state.pending_slider_preview_launch = replay
        worker = self._run_state.simulation_worker
        request_id = replay.request_id
        pending_target_set_ids = list(replay.target_set_ids)
        preview_ownership = getattr(self._run_state, "preview_ownership", None)
        owner_request_id = getattr(preview_ownership, "request_id", None)
        state = self._batch_context_owner.active_batch_state()
        active_fast_parallel = bool(state is not None and state.active and state.parallel and state.fast_mode)
        active_fast_request_id = state.request_id if active_fast_parallel and state is not None else None
        if request_id is not None and owner_request_id is not None and int(request_id) < int(owner_request_id):
            logger.debug(
                "Discarding stale slider simulation request (request_id=%s, preview_owner=%s)",
                request_id,
                owner_request_id,
            )
            self._deps.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        if request_id is None:
            request_id = self._deps.next_slider_preview_request_id()
            self._run_state.pending_slider_preview_launch = replace(
                self._pending_launch(),
                active=True,
                request_id=int(request_id),
                handoff_queued=False,
            )
        self._deps.set_discarded_preview_generation(None)
        self._ui.slider.set_slider_triggered_simulation(True)

        if self._deps.has_active_explicit_simulation() and (
            worker is None or not getattr(worker, "_fast_mode", False)
        ):
            logger.debug("Full simulation in progress; deferring slider update")
            self._queue_current_replay(
                request_id=int(request_id),
                pending_target_set_ids=pending_target_set_ids,
            )
            return

        if bool(getattr(self._run_state, "simulation_running", False)):
            if (
                active_fast_parallel
                and request_id is not None
                and active_fast_request_id is not None
                and int(request_id) != int(active_fast_request_id)
            ):
                logger.debug(
                    "Superseding active fast parallel slider batch (active_request_id=%s, pending_request_id=%s)",
                    active_fast_request_id,
                    request_id,
                )
                self._queue_current_replay(
                    request_id=int(request_id),
                    pending_target_set_ids=pending_target_set_ids,
                )
                supersede_result = self._deps.supersede_parallel_batch_run_soft()
                try:
                    _cancelled, running = supersede_result
                except (TypeError, ValueError):
                    running = 0
                self._run_state.simulation_running = False
                self._run_state.slider_simulation_active = False
                if int(running) > 0:
                    return
                state = self._batch_context_owner.active_batch_state()
            else:
                logger.debug("Simulation already active; deferring slider update")
                self._queue_current_replay(
                    request_id=int(request_id),
                    pending_target_set_ids=pending_target_set_ids,
                )
                return

        if self._deps.worker_is_running(worker):
            logger.debug("Simulation currently running; deferring slider update")
            self._queue_current_replay(
                request_id=int(request_id),
                pending_target_set_ids=pending_target_set_ids,
            )
            return

        active_fast_batch_work = bool(
            state is not None
            and state.fast_mode
            and (state.active or self._deps.has_active_parallel_batch_work())
        )
        if active_fast_batch_work:
            logger.debug("Fast slider run currently running; deferring slider update")
            self._queue_current_replay(
                request_id=int(request_id),
                pending_target_set_ids=pending_target_set_ids,
            )
            return
        self._deps.prune_stopped_owned_simulation_workers()
        if self._deps.has_running_owned_simulation_workers():
            logger.warning(
                "Slider-triggered run blocked while previous simulation worker shutdown remains in progress"
            )
            self._queue_current_replay(
                request_id=int(request_id),
                pending_target_set_ids=pending_target_set_ids,
            )
            self._ui.slider.set_slider_triggered_simulation(False)
            self._run_state.simulation_running = False
            self._run_state.slider_simulation_active = False
            self._ui.run_ui.set_run_button_enabled(True)
            self._ui.run_ui.set_stop_button_enabled(False)
            self._ui.run_ui.set_status_text("Cancelling previous simulation...")
            return

        selected_rows = self._ui.batch.batch_rows_for_scope("selected")
        selected_rows = self._deps.slider_target_rows_for_dispatch(
            selected_rows,
            target_set_ids=pending_target_set_ids,
        )
        if not selected_rows:
            logger.debug(
                "Discarding slider replay launch with no resolvable target rows (target_set_ids=%s)",
                pending_target_set_ids,
            )
            self._ui.slider.set_slider_triggered_simulation(False)
            self._deps.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return

        uses_parallel_batch_runtime = self._deps.slider_preview_uses_parallel_batch_runtime(selected_rows)
        preview_snapshot = self._deps.slider_preview_runtime_snapshot(selected_rows)
        if bool(preview_snapshot.required) and not bool(preview_snapshot.ready):
            if uses_parallel_batch_runtime:
                self._deps.ensure_parallel_batch_pool_eagerly_created(wait=False)
                self._ui.slider.set_slider_triggered_simulation(False)
                self._ui.run_ui.set_runtime_backed_run_controls_ready(False)
                self._ui.run_ui.set_status_text(str(preview_snapshot.message or "Preparing batch runtime..."))
                if bool(preview_snapshot.should_poll):
                    self._ui.run_ui.schedule_runtime_availability_refresh()
                    QtCore.QTimer.singleShot(50, self._deps.retry_slider_preview_launch)
                else:
                    self._deps.clear_pending_slider_preview_replay(clear_plot_updates=False)
                    with suppress(Exception):
                        self._ui.slider.show_preview_unavailable_for_dirty_state(
                            str(preview_snapshot.message or "Batch runtime is not ready.")
                        )
                return
            self._deps.ensure_interactive_simulation_runtime_available_for_mode(
                fast_mode=True,
                wait=False,
            )
            self._ui.slider.set_slider_triggered_simulation(False)
            self._ui.run_ui.set_status_text(str(preview_snapshot.message or "Preparing preview runtime..."))
            if bool(preview_snapshot.should_poll):
                QtCore.QTimer.singleShot(50, self._deps.retry_slider_preview_launch)
            else:
                self._deps.clear_pending_slider_preview_replay(clear_plot_updates=False)
                with suppress(Exception):
                    self._ui.slider.show_preview_unavailable_for_dirty_state(
                        str(preview_snapshot.message or "Preview runtime is not ready.")
                    )
            return

        self._run_state.pending_slider_preview_launch = PendingSliderPreviewLaunchState()

        self._run_state.simulation_running = True
        self._ui.run_ui.set_stop_button_enabled(True)
        self._run_state.slider_simulation_active = True
        request_id = self._deps.mark_request_started(int(request_id))

        logger.info("Starting slider-triggered simulation")
        self._ui.run_ui.set_status_text("Updating simulation...")
        self._ui.run_ui.set_sim_progress_value(0)

        return SliderPreviewLaunchRunRequest(
            request_id=int(request_id),
            batch_rows=tuple(int(row) for row in selected_rows),
            reuse_parallel_lane_pool=True,
        )
