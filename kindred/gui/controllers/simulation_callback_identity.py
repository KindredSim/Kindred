from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SimulationCallbackIdentity:
    run_id: int
    fast_mode: bool
    request_id: int
    preview_owner_epoch: Optional[int]
    batch_set: Optional[str]
    batch_set_id: Optional[str]
    cache_key: str
    callback_context: Mapping[str, Any]
    simulation_identity: Mapping[str, Any]
    preview_batch_cache_token: Optional[str] = None
    launch_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            raise ValueError("SimulationCallbackIdentity.run_id is required.")
        if self.fast_mode is None:
            raise ValueError("SimulationCallbackIdentity.fast_mode is required.")
        if self.request_id is None:
            raise ValueError("SimulationCallbackIdentity.request_id is required.")
        if self.cache_key is None:
            raise ValueError("SimulationCallbackIdentity.cache_key is required.")
        if not isinstance(self.callback_context, Mapping):
            raise ValueError("SimulationCallbackIdentity.callback_context is required.")
        if not isinstance(self.simulation_identity, Mapping):
            raise ValueError("SimulationCallbackIdentity.simulation_identity is required.")
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "fast_mode", bool(self.fast_mode))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(
            self,
            "preview_owner_epoch",
            int(self.preview_owner_epoch) if self.preview_owner_epoch is not None else None,
        )
        object.__setattr__(self, "batch_set", str(self.batch_set) if self.batch_set is not None else None)
        object.__setattr__(self, "batch_set_id", str(self.batch_set_id) if self.batch_set_id is not None else None)
        object.__setattr__(self, "cache_key", str(self.cache_key))
        object.__setattr__(self, "callback_context", self.callback_context)
        object.__setattr__(self, "simulation_identity", dict(self.simulation_identity))
        object.__setattr__(
            self,
            "preview_batch_cache_token",
            str(self.preview_batch_cache_token) if self.preview_batch_cache_token is not None else None,
        )
        object.__setattr__(
            self,
            "launch_provenance",
            dict(self.launch_provenance) if isinstance(self.launch_provenance, Mapping) else None,
        )

    @classmethod
    def capture(
        cls,
        *,
        run_id: int,
        fast_mode: bool,
        request_id: int,
        preview_owner_epoch: Optional[int],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        cache_key: str,
        callback_context: Mapping[str, Any],
        simulation_identity: Mapping[str, Any],
        preview_batch_cache_token: Optional[str] = None,
        launch_provenance: Mapping[str, Any] | None = None,
    ) -> "SimulationCallbackIdentity":
        if run_id is None:
            raise ValueError("SimulationCallbackIdentity.run_id is required.")
        if fast_mode is None:
            raise ValueError("SimulationCallbackIdentity.fast_mode is required.")
        if request_id is None:
            raise ValueError("SimulationCallbackIdentity.request_id is required.")
        if cache_key is None:
            raise ValueError("SimulationCallbackIdentity.cache_key is required.")
        return cls(
            run_id=int(run_id),
            fast_mode=bool(fast_mode),
            request_id=int(request_id),
            preview_owner_epoch=int(preview_owner_epoch) if preview_owner_epoch is not None else None,
            batch_set=str(batch_set) if batch_set is not None else None,
            batch_set_id=str(batch_set_id) if batch_set_id is not None else None,
            cache_key=str(cache_key),
            callback_context=callback_context,
            simulation_identity=simulation_identity,
            preview_batch_cache_token=(
                str(preview_batch_cache_token) if preview_batch_cache_token is not None else None
            ),
            launch_provenance=launch_provenance,
        )
