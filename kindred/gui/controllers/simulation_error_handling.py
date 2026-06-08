from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_failure_policy import (
    SimulationFailureDecision,
    SimulationFailurePolicyOwner,
)


@dataclass(frozen=True)
class SimulationErrorHandlingDependencies:
    apply_failure_decision: Callable[[SimulationFailureDecision], None]


class SimulationErrorHandlingOwner:
    """Adapts direct worker error callbacks into the canonical failure policy path."""

    def __init__(
        self,
        *,
        failure_policy_owner: SimulationFailurePolicyOwner,
        dependencies: SimulationErrorHandlingDependencies,
    ) -> None:
        self._failure_policy_owner = failure_policy_owner
        self._deps = dependencies

    def handle_error(
        self,
        error_msg: object,
        *,
        callback_identity: SimulationCallbackIdentity,
    ) -> None:
        decision = self._failure_policy_owner.resolve_direct_error(
            error_msg,
            callback_identity=callback_identity,
        )
        self._deps.apply_failure_decision(decision)
