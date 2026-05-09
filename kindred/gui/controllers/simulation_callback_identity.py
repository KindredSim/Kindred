from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SimulationCallbackIdentity:
    run_id: Optional[int]
    fast_mode: Optional[bool]
    request_id: Optional[int]
    owner_epoch: Optional[int]
    batch_set: Optional[str]
    batch_set_id: Optional[str]
    cache_key: Optional[str]
    callback_context: Mapping[str, Any] | None = None
    simulation_identity: Mapping[str, Any] | None = None
    preview_batch_cache_token: Optional[str] = None

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
        callback_context: Mapping[str, Any] | None = None,
        simulation_identity: Mapping[str, Any] | None = None,
        preview_batch_cache_token: Optional[str] = None,
    ) -> "SimulationCallbackIdentity":
        return cls(
            run_id=int(run_id) if run_id is not None else None,
            fast_mode=bool(fast_mode) if fast_mode is not None else None,
            request_id=int(request_id) if request_id is not None else None,
            owner_epoch=int(owner_epoch) if owner_epoch is not None else None,
            batch_set=str(batch_set) if batch_set is not None else None,
            batch_set_id=str(batch_set_id) if batch_set_id is not None else None,
            cache_key=str(cache_key) if cache_key is not None else None,
            callback_context=callback_context if isinstance(callback_context, Mapping) else None,
            simulation_identity=dict(simulation_identity) if isinstance(simulation_identity, Mapping) else None,
            preview_batch_cache_token=(
                str(preview_batch_cache_token) if preview_batch_cache_token is not None else None
            ),
        )
