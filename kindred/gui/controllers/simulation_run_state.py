from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore


class SimulationRunState(QtCore.QObject):
    """Owns per-run worker, request-id, and progress-throttle state."""

    def __init__(self, *, on_progress_timeout, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self.simulation_running = False
        self.simulation_worker = None
        self.processing_progress = False
        self.pending_progress_payload: Optional[Tuple[int, str]] = None
        self.progress_flush_interval_ms = 33
        self.progress_flush_timer = QtCore.QTimer(self)
        self.progress_flush_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.progress_flush_timer.setInterval(int(self.progress_flush_interval_ms))
        self.progress_flush_timer.timeout.connect(on_progress_timeout)
        self.slider_simulation_active = False
        self.pending_slider_simulation = False
        self.run_sequence_id = 0
        self.active_run_id = 0
        self.sim_request_id = 0
        self.latest_sim_request_id = 0
        self.pending_slider_sim_request_id: Optional[int] = None
        self.pending_slider_target_set_ids: Tuple[str, ...] = ()

    def next_request_id(self) -> int:
        self.sim_request_id = int(self.sim_request_id) + 1
        self.latest_sim_request_id = int(self.sim_request_id)
        return int(self.sim_request_id)

    def reserve_request_id(self) -> int:
        self.sim_request_id = int(self.sim_request_id) + 1
        return int(self.sim_request_id)
