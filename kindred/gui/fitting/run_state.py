from __future__ import annotations

from typing import Sequence


class FittingRunStateOwner:
    """Owns mutable per-fit run identity and active-result state."""

    def __init__(self) -> None:
        self._active_dataset_ids: tuple[str, ...] = ()
        self._active_run_stamp_hash = ""
        self._active_run_superseded = False

    @property
    def active_dataset_ids(self) -> tuple[str, ...]:
        return tuple(self._active_dataset_ids)

    def set_active_dataset_ids(self, dataset_ids: Sequence[str]) -> None:
        self._active_dataset_ids = tuple(str(dataset_id) for dataset_id in dataset_ids if str(dataset_id))

    def clear_active_dataset_ids(self) -> None:
        self._active_dataset_ids = ()

    @property
    def active_run_stamp_hash(self) -> str:
        return str(self._active_run_stamp_hash or "")

    @property
    def active_run_superseded(self) -> bool:
        return bool(self._active_run_superseded)

    def set_active_run_stamp_hash(self, stamp_hash: str) -> None:
        self._active_run_stamp_hash = str(stamp_hash or "")
        self._active_run_superseded = False

    def clear_active_run_stamp_hash(self) -> None:
        self._active_run_stamp_hash = ""

    def mark_superseded(self) -> None:
        self._active_run_superseded = True
        self._active_run_stamp_hash = ""

    def clear_superseded(self) -> None:
        self._active_run_superseded = False

    def reset_for_new_run(self) -> None:
        self._active_dataset_ids = ()
        self._active_run_stamp_hash = ""
        self._active_run_superseded = False
