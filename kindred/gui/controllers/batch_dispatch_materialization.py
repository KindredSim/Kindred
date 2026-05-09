from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from kindred.gui.controllers.simulation_completion_policy import pending_initial_seed_for_set


def _try_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


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
        pending_init_seed: Mapping[str, Mapping[str, Any]] | None,
        pending_init_applied: bool,
    ) -> dict[str, float]:
        initials = dict(self._batch.batch_initials_for_row(int(row)))
        pending_seed_for_set = pending_initial_seed_for_set(
            pending_init_seed,
            set_name=str(set_name),
        )
        if pending_seed_for_set and (not bool(pending_init_applied)):
            for species, value in pending_seed_for_set.items():
                float_value = _try_float(value)
                if float_value is None:
                    continue
                initials[str(species)] = float_value
        if bool(fast_mode):
            initials = dict(self._slider.preview_initials_for_row(int(row), initials))
        return {str(species): float(value) for species, value in initials.items()}
