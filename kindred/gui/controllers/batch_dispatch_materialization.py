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
        fast_mode: bool,
        pending_initials: dict[str, float] | None = None,
    ) -> dict[str, float]:
        initials = dict(self._batch.batch_initials_for_row(int(row)))
        for species, value in dict(pending_initials or {}).items():
            initials[str(species)] = float(value)
        if bool(fast_mode):
            initials = dict(self._slider.preview_initials_for_row(int(row), initials))
        return {str(species): float(value) for species, value in initials.items()}
