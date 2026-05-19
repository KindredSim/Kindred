from __future__ import annotations

from typing import Any

from kindred.gui.controllers.batch_run_context_owner import (
    BatchContextSeed,
    BatchRunContextOwner,
)
from kindred.gui.ports import CompletedRunDisplayIntent


def seed_batch_context(owner: BatchRunContextOwner, **kwargs: Any) -> None:
    if kwargs.get("completed_run_display_intent") is None:
        queue_ids = tuple(str(set_id) for set_id in (kwargs.get("queue_ids") or ()) if str(set_id))
        queue_names = tuple(str(name) for name in (kwargs.get("queue_names") or ()))
        if queue_ids:
            primary_set_id = str(kwargs.get("primary_set_id") or queue_ids[0])
            if primary_set_id not in queue_ids:
                primary_set_id = queue_ids[0]
            kwargs["completed_run_display_intent"] = CompletedRunDisplayIntent(
                set_ids=queue_ids,
                labels_by_set_id={
                    str(set_id): (
                        str(queue_names[index])
                        if index < len(queue_names) and str(queue_names[index])
                        else str(set_id)
                    )
                    for index, set_id in enumerate(queue_ids)
                },
                primary_set_id=primary_set_id,
                cache_key=str(kwargs.get("cache_key") or ""),
                run_id=kwargs.get("run_id"),
                request_id=kwargs.get("request_id"),
            )
    owner.load_context(BatchContextSeed(**kwargs))
