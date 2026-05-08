from __future__ import annotations

from typing import Any

from kindred.gui.controllers.batch_run_context_owner import (
    BatchContextSeed,
    BatchRunContextOwner,
)


def seed_batch_context(owner: BatchRunContextOwner, **kwargs: Any) -> None:
    owner.load_context(BatchContextSeed(**kwargs))
