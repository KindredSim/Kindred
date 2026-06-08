from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Set

from PySide6 import QtCore

from kindred.gui.ports import FreshPreviewDisplayEntry


@dataclass
class PendingSliderPlotUpdate:
    set_ids: Set[str] = field(default_factory=set)
    target_set_ids: tuple[str, ...] = ()
    cache_key: Optional[str] = None
    cache_kind: Optional[str] = None
    request_id: Optional[int] = None
    run_id: Optional[int] = None
    accepted_preview_request_id: Optional[int] = None
    accepted_preview_owner_epoch: Optional[int] = None
    valid_set_ids: Optional[tuple[str, ...]] = None
    fresh_preview_entries: Dict[str, FreshPreviewDisplayEntry] = field(default_factory=dict)


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

    def requeue_pending(
        self,
        pending: PendingSliderPlotUpdate,
        *,
        interval_ms: Optional[int] = None,
    ) -> None:
        """Restore a pending update for a later flush.

        Preview display is only valid once all targeted fresh-preview entries
        have arrived.  The controller uses this to avoid publishing an older
        cached display for a partially completed preview batch.
        """
        if self.timer.isActive():
            self.timer.stop()
        self.pending = PendingSliderPlotUpdate(
            set_ids=set(pending.set_ids),
            target_set_ids=tuple(pending.target_set_ids or ()),
            cache_key=pending.cache_key,
            cache_kind=pending.cache_kind,
            request_id=pending.request_id,
            run_id=pending.run_id,
            accepted_preview_request_id=pending.accepted_preview_request_id,
            accepted_preview_owner_epoch=pending.accepted_preview_owner_epoch,
            valid_set_ids=pending.valid_set_ids,
            fresh_preview_entries=dict(pending.fresh_preview_entries),
        )
        delay_ms = self.slider_interval_ms if interval_ms is None else int(interval_ms)
        try:
            self.timer.setInterval(max(1, int(delay_ms)))
        except Exception:
            self.timer.setInterval(max(1, int(self.slider_interval_ms)))
        self.timer.start()

    def queue(
        self,
        *,
        set_id: Optional[str],
        cache_key: Optional[str],
        request_id: Optional[int],
        run_id: Optional[int],
        slider_triggered: bool,
        preview_request_id: Optional[int],
        preview_owner_epoch: Optional[int],
        preview_target_set_ids: Sequence[str],
        latest_request_id: int,
        valid_set_ids: Optional[Sequence[str]],
        fresh_preview_entry: Optional[FreshPreviewDisplayEntry],
        active_run_id: int,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        cache_token = str(cache_key or "").strip()
        if not cache_token:
            return
        request_accepted = self._request_can_display(
            request_id=request_id,
            slider_triggered=bool(slider_triggered),
            preview_request_id=preview_request_id,
            latest_request_id=int(latest_request_id),
        )
        if request_id is not None and not bool(request_accepted):
            return
        if run_id is not None and int(run_id) != int(active_run_id):
            return
        incoming_cache_kind = "preview" if bool(slider_triggered) else "result"
        normalized_targets = (
            tuple(str(set_id) for set_id in (preview_target_set_ids or ()) if str(set_id))
            if bool(slider_triggered) and bool(request_accepted)
            else ()
        )
        accepted_preview_request_id = (
            int(preview_request_id)
            if bool(slider_triggered)
            and bool(request_accepted)
            and preview_request_id is not None
            else None
        )
        accepted_preview_owner_epoch = (
            int(preview_owner_epoch)
            if bool(slider_triggered)
            and bool(request_accepted)
            and preview_owner_epoch is not None
            else None
        )
        if self.pending.set_ids and incoming_cache_kind == "preview":
            pending_owner_key = (
                self.pending.request_id,
                self.pending.accepted_preview_request_id,
                self.pending.accepted_preview_owner_epoch,
                self.pending.cache_key,
                self.pending.run_id,
                self.pending.target_set_ids,
            )
            incoming_owner_key = (
                int(request_id) if request_id is not None else None,
                int(accepted_preview_request_id) if accepted_preview_request_id is not None else None,
                int(accepted_preview_owner_epoch) if accepted_preview_owner_epoch is not None else None,
                cache_token,
                int(run_id) if run_id is not None else None,
                normalized_targets,
            )
            if pending_owner_key != incoming_owner_key:
                self.pending = PendingSliderPlotUpdate()
        sid = str(set_id or "").strip()
        if sid:
            self.pending.set_ids.add(sid)
        if fresh_preview_entry is not None:
            fresh_set_id = str(fresh_preview_entry.set_id or sid).strip()
            if fresh_set_id:
                self.pending.fresh_preview_entries[fresh_set_id] = fresh_preview_entry
        if incoming_cache_kind == "preview":
            self.pending.target_set_ids = normalized_targets
        else:
            self.pending.target_set_ids = ()
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

    @staticmethod
    def _request_can_display(
        *,
        request_id: Optional[int],
        slider_triggered: bool,
        preview_request_id: Optional[int],
        latest_request_id: int,
    ) -> bool:
        if request_id is None:
            return True
        if bool(slider_triggered):
            return preview_request_id is not None and int(preview_request_id) == int(request_id)
        return int(request_id) == int(latest_request_id)

    def take_pending(self) -> PendingSliderPlotUpdate:
        if self.timer.isActive():
            self.timer.stop()
        pending = PendingSliderPlotUpdate(
            set_ids=set(self.pending.set_ids),
            target_set_ids=self.pending.target_set_ids,
            cache_key=self.pending.cache_key,
            cache_kind=self.pending.cache_kind,
            request_id=self.pending.request_id,
            run_id=self.pending.run_id,
            accepted_preview_request_id=self.pending.accepted_preview_request_id,
            accepted_preview_owner_epoch=self.pending.accepted_preview_owner_epoch,
            valid_set_ids=self.pending.valid_set_ids,
            fresh_preview_entries=dict(self.pending.fresh_preview_entries),
        )
        self.pending = PendingSliderPlotUpdate()
        return pending
