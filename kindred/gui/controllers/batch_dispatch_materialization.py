from __future__ import annotations

from typing import Any

class BatchDispatchMaterializationOwner:
    def __init__(self, *, batch: Any, slider: Any) -> None:
        self._batch = batch
        self._slider = slider

    def materialize_initials(
        self,
        *,
        row: int,
        set_name: str,
        fast_mode: bool,
    ) -> dict[str, float]:
        initials = dict(self._batch.batch_initials_for_row(int(row)))
        if bool(fast_mode):
            initials = dict(self._slider.preview_initials_for_row(int(row), initials))
        return {str(species): float(value) for species, value in initials.items()}
