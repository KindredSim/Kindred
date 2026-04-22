from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from PySide6 import QtCore


def _normalize_preview_request_id(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_preview_epoch(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _normalize_preview_target_set_ids(values: Sequence[str] | object) -> tuple[str, ...]:
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
class PreviewOwnershipState:
    request_id: Optional[int] = None
    epoch: int = 0
    target_set_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _normalize_preview_request_id(self.request_id))
        object.__setattr__(self, "epoch", _normalize_preview_epoch(self.epoch))
        object.__setattr__(
            self,
            "target_set_ids",
            _normalize_preview_target_set_ids(self.target_set_ids),
        )


class SimulationRunState(QtCore.QObject):
    """Owns per-run worker, request-id, progress-throttle, and preview-ownership state."""

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
        self.preview_ownership = PreviewOwnershipState()

    def next_request_id(self) -> int:
        self.sim_request_id = int(self.sim_request_id) + 1
        self.latest_sim_request_id = int(self.sim_request_id)
        return int(self.sim_request_id)

    def reserve_request_id(self) -> int:
        self.sim_request_id = int(self.sim_request_id) + 1
        return int(self.sim_request_id)
