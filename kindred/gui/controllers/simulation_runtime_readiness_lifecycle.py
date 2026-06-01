from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationRuntimeReadinessRenderState:
    status: str
    status_text: str = ""
    launch_available: bool = False
    preview_available: bool = False
    failed: bool = False
    retryable: bool = False
    clear_status: bool = False
    preview_unavailable_status: str = ""


@dataclass(frozen=True)
class SimulationRuntimeReadinessEndpointState:
    manual_retry_available: bool = False
    backend_warmup_pending: bool = False
    pending_intent_kind: str = ""
