from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping
from typing import Any, Optional

from kindred.core.containment_kernel import (
    ContainmentHandlerSpec,
    ContainmentKernelEvent,
    ContainmentKernelOwner,
)


class SimulationRuntimeOwner:
    """Non-GUI owner facade for simulation containment lifecycle."""

    def __init__(
        self,
        *,
        handler_import_path: str,
        startup_payload: Mapping[str, Any] | None = None,
        handler_env: Mapping[str, str] | None = None,
        ready_timeout_s: float = 30.0,
        accept_timeout_s: float = 10.0,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        self._kernel_owner = ContainmentKernelOwner(
            ContainmentHandlerSpec(import_path=str(handler_import_path), env=dict(handler_env or {})),
            startup_payload=dict(startup_payload or {}),
            ready_timeout_s=float(ready_timeout_s),
            accept_timeout_s=float(accept_timeout_s),
            mp_context=mp_context,
        )

    @property
    def owner_epoch(self) -> int:
        return int(self._kernel_owner.owner_epoch)

    @property
    def startup_payload(self) -> dict[str, Any]:
        return self._kernel_owner.startup_payload

    @property
    def is_running(self) -> bool:
        return bool(self._kernel_owner.is_running)

    def drain_events(self) -> list[ContainmentKernelEvent]:
        return self._kernel_owner.drain_events()

    def warm(
        self,
        *,
        wait: bool = False,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._kernel_owner.start(wait=bool(wait), cancellation_check=cancellation_check)

    def solve(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        active_timeout_s: float = 60.0,
        cancellation_check: Optional[Callable[[], bool]] = None,
        reply_fields: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._kernel_owner.request(
            dict(payload or {}),
            active_timeout_s=float(active_timeout_s),
            cancellation_check=cancellation_check,
            reply_fields=dict(reply_fields or {}),
        )

    def close(self, *, kill: bool = False) -> None:
        self._kernel_owner.close(kill=bool(kill))
