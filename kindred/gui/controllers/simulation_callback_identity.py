from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Mapping, Optional

from kindred.gui.controllers.simulation_completion_policy import CompletionPolicyContext


@dataclass(frozen=True)
class SimulationCallbackIdentity:
    run_id: Optional[int]
    fast_mode: Optional[bool]
    request_id: Optional[int]
    owner_epoch: Optional[int]
    batch_set: Optional[str]
    batch_set_id: Optional[str]
    cache_key: Optional[str]
    policy_context: CompletionPolicyContext | None = None
    context_snapshot: Mapping[str, Any] | None = None

    @classmethod
    def capture(
        cls,
        *,
        run_id: Optional[int],
        fast_mode: Optional[bool],
        request_id: Optional[int],
        owner_epoch: Optional[int],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        cache_key: Optional[str],
        policy_context: CompletionPolicyContext | None = None,
        context_snapshot: Mapping[str, Any] | None = None,
    ) -> "SimulationCallbackIdentity":
        return cls(
            run_id=int(run_id) if run_id is not None else None,
            fast_mode=bool(fast_mode) if fast_mode is not None else None,
            request_id=int(request_id) if request_id is not None else None,
            owner_epoch=int(owner_epoch) if owner_epoch is not None else None,
            batch_set=str(batch_set) if batch_set is not None else None,
            batch_set_id=str(batch_set_id) if batch_set_id is not None else None,
            cache_key=str(cache_key) if cache_key is not None else None,
            policy_context=policy_context,
            context_snapshot=deepcopy(dict(context_snapshot)) if isinstance(context_snapshot, Mapping) else None,
        )
