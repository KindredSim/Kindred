from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Set

from PySide6 import QtCore


@dataclass
class PendingSliderPlotUpdate:
    set_ids: Set[str] = field(default_factory=set)
    cache_key: Optional[str] = None
    cache_kind: Optional[str] = None
    request_id: Optional[int] = None
    run_id: Optional[int] = None
    accepted_preview_request_id: Optional[int] = None
    accepted_preview_owner_epoch: Optional[int] = None
    valid_set_ids: Optional[tuple[str, ...]] = None
    allow_fallback: bool = True


class SliderPlotCoalescer(QtCore.QObject):
    """Owns pending cache-backed plot-update state and coalescing timer wiring."""

    def __init__(
        self,
        *,
        on_timeout,
        parent: QtCore.QObject,
        slider_interval_ms: int = 24,
        explicit_interval_ms: int = 90,
    ) -> None:
        super().__init__(parent)
        self.slider_interval_ms = int(slider_interval_ms)
        self.explicit_interval_ms = int(explicit_interval_ms)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(int(self.slider_interval_ms))
        self.timer.timeout.connect(on_timeout)
        self.pending = PendingSliderPlotUpdate()

    def clear(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.pending = PendingSliderPlotUpdate()

    def queue(
        self,
        *,
        set_id: Optional[str],
        cache_key: Optional[str],
        request_id: Optional[int],
        request_accepted: bool,
        run_id: Optional[int],
        accepted_preview_request_id: Optional[int],
        accepted_preview_owner_epoch: Optional[int],
        slider_triggered: bool,
        valid_set_ids: Optional[Sequence[str]],
        allow_fallback: bool,
        active_run_id: int,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        cache_token = str(cache_key or "").strip()
        if not cache_token:
            return
        if request_id is not None and not bool(request_accepted):
            return
        if run_id is not None and int(run_id) != int(active_run_id):
            return
        incoming_cache_kind = "preview" if bool(slider_triggered) else "result"
        if self.pending.set_ids and incoming_cache_kind == "preview":
            pending_owner_key = (
                self.pending.request_id,
                self.pending.accepted_preview_request_id,
                self.pending.accepted_preview_owner_epoch,
                self.pending.cache_key,
                self.pending.run_id,
            )
            incoming_owner_key = (
                int(request_id) if request_id is not None else None,
                int(accepted_preview_request_id) if accepted_preview_request_id is not None else None,
                int(accepted_preview_owner_epoch) if accepted_preview_owner_epoch is not None else None,
                cache_token,
                int(run_id) if run_id is not None else None,
            )
            if pending_owner_key != incoming_owner_key:
                self.pending = PendingSliderPlotUpdate()
        sid = str(set_id or "").strip()
        if sid:
            self.pending.set_ids.add(sid)
        self.pending.cache_key = cache_token
        self.pending.cache_kind = incoming_cache_kind
        self.pending.request_id = int(request_id) if request_id is not None else None
        self.pending.run_id = int(run_id) if run_id is not None else None
        self.pending.accepted_preview_request_id = (
            int(accepted_preview_request_id) if accepted_preview_request_id is not None else None
        )
        self.pending.accepted_preview_owner_epoch = (
            int(accepted_preview_owner_epoch) if accepted_preview_owner_epoch is not None else None
        )
        self.pending.valid_set_ids = (
            tuple(str(set_id) for set_id in valid_set_ids)
            if valid_set_ids is not None
            else None
        )
        self.pending.allow_fallback = bool(allow_fallback)
        interval_ms = self.slider_interval_ms if bool(slider_triggered) else self.explicit_interval_ms
        try:
            self.timer.setInterval(max(1, int(interval_ms)))
        except Exception as exc:
            record_nonfatal_exception(
                f"Failed to set slider plot coalesce timer interval_ms={int(interval_ms)}",
                exc,
            )
        if not self.timer.isActive():
            self.timer.start()

    def take_pending(self) -> PendingSliderPlotUpdate:
        if self.timer.isActive():
            self.timer.stop()
        pending = PendingSliderPlotUpdate(
            set_ids=set(self.pending.set_ids),
            cache_key=self.pending.cache_key,
            cache_kind=self.pending.cache_kind,
            request_id=self.pending.request_id,
            run_id=self.pending.run_id,
            accepted_preview_request_id=self.pending.accepted_preview_request_id,
            accepted_preview_owner_epoch=self.pending.accepted_preview_owner_epoch,
            valid_set_ids=self.pending.valid_set_ids,
            allow_fallback=self.pending.allow_fallback,
        )
        self.pending = PendingSliderPlotUpdate()
        return pending
