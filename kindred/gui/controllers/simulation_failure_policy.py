from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from kindred.core.simulation_failure import (
    build_simulation_failure,
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_completion_policy import (
    CompletionPolicyContext,
    PolicyStatePatch,
    SimulationCompletionPolicy,
)
from kindred.gui.controllers.preview_target_identity import normalize_preview_target_set_ids


@dataclass(frozen=True, slots=True)
class SimulationFailureEnvelope:
    payload: Mapping[str, Any]
    kind: str
    code: str | None
    message: str
    user_message: str
    detail_text: str
    details: Mapping[str, Any]
    context: Mapping[str, Any] | None
    exc_type: str | None
    cancelled: bool
    origin: str
    callback_identity: SimulationCallbackIdentity | None = None
    set_id: str = ""
    set_name: str = ""

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        origin: str,
        callback_identity: SimulationCallbackIdentity | None = None,
        set_id: str = "",
        set_name: str = "",
    ) -> "SimulationFailureEnvelope":
        coerced = coerce_simulation_failure(payload)
        details = coerced.get("details") if isinstance(coerced.get("details"), Mapping) else {}
        context = coerced.get("context") if isinstance(coerced.get("context"), Mapping) else None
        return cls(
            payload=coerced,
            kind=str(coerced.get("kind") or "simulation_error"),
            code=str(coerced.get("code")) if coerced.get("code") is not None else None,
            message=str(coerced.get("message") or ""),
            user_message=simulation_failure_user_message(coerced),
            detail_text=simulation_failure_detail_text(coerced),
            details=dict(details or {}),
            context=dict(context) if isinstance(context, Mapping) else None,
            exc_type=str(coerced.get("exc_type")) if coerced.get("exc_type") is not None else None,
            cancelled=bool(is_cancelled_failure(coerced)),
            origin=str(origin or "unknown"),
            callback_identity=callback_identity,
            set_id=str(set_id or ""),
            set_name=str(set_name or set_id or ""),
        )


@dataclass(frozen=True, slots=True)
class SimulationFailureInput:
    error_payload: object
    origin: str
    callback_identity: SimulationCallbackIdentity | None = None
    set_id: str = ""
    set_name: str = ""
    source: str = ""
    context: Mapping[str, Any] | None = None
    active_fast_mode: bool | None = None
    allow_scoped_failure: bool = False
    stale_runtime_lane_outcome: bool = False
    runtime_session_stale: Mapping[str, Any] | None = None
    missing_context_reason: str = ""


@dataclass(frozen=True, slots=True)
class FailureContextMutationRequest:
    kind: str = "none"
    context: Mapping[str, Any] | None = None
    base_context: Mapping[str, Any] | None = None
    set_id: str = ""
    set_name: str = ""
    failure_payload: Mapping[str, Any] | None = None
    state_patch: PolicyStatePatch | None = None

    @classmethod
    def none(cls) -> "FailureContextMutationRequest":
        return cls()


@dataclass(frozen=True, slots=True)
class FailureRuntimeConsequenceRequest:
    kind: str = "none"
    backend_failed: bool = False
    replay_fast_mode: bool = False
    terminal_replay_needed: bool = False
    stale_fast_error_replay_needed: bool = False

    @classmethod
    def none(cls) -> "FailureRuntimeConsequenceRequest":
        return cls()


@dataclass(frozen=True, slots=True)
class FailureUiConsequenceRequest:
    kind: str = "none"
    status_text: str = ""
    progress_label: str = ""
    completed: int = 0
    total: int = 0

    @classmethod
    def none(cls) -> "FailureUiConsequenceRequest":
        return cls()


@dataclass(frozen=True, slots=True)
class FailureDisplayConsequenceRequest:
    kind: str = "none"
    target_set_ids: tuple[str, ...] = ()
    request_id: int | None = None
    run_id: int | None = None
    preview_owner_epoch: int | None = None
    status_text: str = ""
    failure_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_set_ids",
            normalize_preview_target_set_ids(self.target_set_ids),
        )
        if self.request_id is not None:
            object.__setattr__(self, "request_id", int(self.request_id))
        if self.run_id is not None:
            object.__setattr__(self, "run_id", int(self.run_id))
        if self.preview_owner_epoch is not None:
            object.__setattr__(self, "preview_owner_epoch", int(self.preview_owner_epoch))
        object.__setattr__(self, "status_text", str(self.status_text or ""))
        object.__setattr__(
            self,
            "failure_payload",
            dict(self.failure_payload) if isinstance(self.failure_payload, Mapping) else None,
        )

    @classmethod
    def none(cls) -> "FailureDisplayConsequenceRequest":
        return cls()


@dataclass(frozen=True, slots=True)
class SimulationFailureDecision:
    disposition: str
    envelope: SimulationFailureEnvelope | None = None
    callback_identity: SimulationCallbackIdentity | None = None
    freshness: Any | None = None
    matched_context: Mapping[str, Any] | None = None
    context_mutation: FailureContextMutationRequest = field(default_factory=FailureContextMutationRequest.none)
    runtime_consequence: FailureRuntimeConsequenceRequest = field(default_factory=FailureRuntimeConsequenceRequest.none)
    ui_consequence: FailureUiConsequenceRequest = field(default_factory=FailureUiConsequenceRequest.none)
    display_consequence: FailureDisplayConsequenceRequest = field(default_factory=FailureDisplayConsequenceRequest.none)
    accepted_runtime_completion: bool = False
    consumed_runtime_completion: bool = True
    terminal_runtime_completion: bool = False
    failed_runtime_completion: bool = False
    stop_current_poll_batch: bool = False
    log_message: str = ""

    @property
    def terminal_failure(self) -> bool:
        return self.disposition == "terminal_failure"

    @property
    def status_only_preview_failure(self) -> bool:
        return self.disposition == "status_only_preview_failure"

    @property
    def superseded_fast_failure(self) -> bool:
        return self.disposition == "superseded_fast_failure"

    @property
    def scoped_failure(self) -> bool:
        return self.disposition in {"scoped_failure_progress", "scoped_failure_final"}


class SimulationFailurePolicyOwner:
    """Canonical runtime/preview simulation-failure policy owner.

    Direct worker error callbacks and runtime-lane completion failures must both
    enter here after source-specific input adaptation.  This owner decides the
    domain disposition and emits typed consequence requests; it does not mutate
    runtime, UI, dialogs, caches, or batch context directly.
    """

    def __init__(
        self,
        *,
        batch_context_owner: Any,
        completion_policy: SimulationCompletionPolicy,
        freshness: Any,
        completion_policy_preview_ownership: Any,
    ) -> None:
        self._batch_context_owner = batch_context_owner
        self._completion_policy = completion_policy
        self._freshness = freshness
        self._completion_policy_preview_ownership = completion_policy_preview_ownership

    def resolve_direct_error(
        self,
        error_payload: object,
        *,
        callback_identity: SimulationCallbackIdentity,
    ) -> SimulationFailureDecision:
        return self.resolve_failure(
            SimulationFailureInput(
                error_payload=error_payload,
                origin="direct_worker",
                callback_identity=callback_identity,
                set_id=str(callback_identity.batch_set_id or ""),
                set_name=str(callback_identity.batch_set or callback_identity.batch_set_id or ""),
                active_fast_mode=bool(callback_identity.fast_mode),
                allow_scoped_failure=False,
            )
        )

    def resolve_failure(self, failure_input: SimulationFailureInput) -> SimulationFailureDecision:
        if bool(failure_input.stale_runtime_lane_outcome):
            return self._resolve_stale_runtime_lane_outcome(failure_input)

        callback_identity = failure_input.callback_identity
        if callback_identity is None:
            payload = failure_input.error_payload or build_simulation_failure(
                "runtime_missing_callback_identity",
                "Missing callback identity for active parallel batch outcome.",
            )
            return self._terminal_decision(
                failure_input,
                payload=payload,
                callback_identity=None,
                context=None,
                freshness=None,
                fast_mode=bool(failure_input.active_fast_mode),
                missing_context_reason="missing-callback-identity",
            )

        context = failure_input.context if isinstance(failure_input.context, Mapping) else None
        context_reason = str(failure_input.missing_context_reason or "")
        if context is None:
            context_resolution = self._batch_context_owner.context_for_callback_identity(callback_identity)
            if bool(getattr(context_resolution, "matched", False)) and isinstance(
                getattr(context_resolution, "context", None),
                Mapping,
            ):
                context = getattr(context_resolution, "context")
            else:
                context_reason = str(getattr(context_resolution, "reason", "missing-context") or "missing-context")

        freshness = self._freshness.assess_callback(callback_identity, context=context if isinstance(context, Mapping) else None)
        envelope = SimulationFailureEnvelope.from_payload(
            failure_input.error_payload,
            origin=failure_input.origin,
            callback_identity=callback_identity,
            set_id=failure_input.set_id or callback_identity.batch_set_id or "",
            set_name=failure_input.set_name or callback_identity.batch_set or callback_identity.batch_set_id or "",
        )

        if bool(getattr(freshness, "stale_run", False)):
            return SimulationFailureDecision(
                disposition="ignore_stale_run",
                envelope=envelope,
                callback_identity=callback_identity,
                freshness=freshness,
                matched_context=context,
                consumed_runtime_completion=False,
                log_message=envelope.user_message,
            )

        if not isinstance(context, Mapping):
            if bool(getattr(freshness, "superseded_fast_request", False)):
                return self._superseded_fast_decision(
                    failure_input,
                    envelope=envelope,
                    callback_identity=callback_identity,
                    context=None,
                    policy_context=None,
                    freshness=freshness,
                )
            missing_payload = build_simulation_failure(
                "missing_failure_context",
                f"Simulation failure requires a current batch context ({context_reason or 'missing-context'}).",
                details={"original_failure": dict(envelope.payload), "origin": str(failure_input.origin or "")},
            )
            return self._terminal_decision(
                failure_input,
                payload=missing_payload,
                callback_identity=callback_identity,
                context=None,
                freshness=freshness,
                fast_mode=bool(callback_identity.fast_mode),
                missing_context_reason=context_reason,
            )

        if bool(getattr(freshness, "dispatch_identity_stale", False)):
            return SimulationFailureDecision(
                disposition="ignore_stale_dispatch_identity",
                envelope=envelope,
                callback_identity=callback_identity,
                freshness=freshness,
                matched_context=context,
                context_mutation=FailureContextMutationRequest(
                    kind="mark_stale_dispatch_identity_consumed",
                    context=context,
                    set_id=failure_input.set_id or callback_identity.batch_set_id or "",
                ),
                consumed_runtime_completion=True,
                log_message=envelope.user_message,
            )

        if bool(getattr(freshness, "runtime_input_stale", False)):
            return SimulationFailureDecision(
                disposition="ignore_stale_runtime_input",
                envelope=envelope,
                callback_identity=callback_identity,
                freshness=freshness,
                matched_context=context,
                context_mutation=FailureContextMutationRequest(
                    kind="mark_stale_runtime_input_consumed",
                    context=context,
                    set_id=failure_input.set_id or callback_identity.batch_set_id or "",
                ),
                consumed_runtime_completion=True,
                log_message=envelope.user_message,
            )

        policy_context = self._batch_context_owner.completion_policy_context(context)
        if bool(getattr(freshness, "superseded_fast_request", False)):
            return self._superseded_fast_decision(
                failure_input,
                envelope=envelope,
                callback_identity=callback_identity,
                context=context,
                policy_context=policy_context,
                freshness=freshness,
            )

        if bool(failure_input.allow_scoped_failure):
            scoped = self._scoped_failure_decision(
                failure_input,
                envelope=envelope,
                callback_identity=callback_identity,
                context=context,
                freshness=freshness,
            )
            if scoped is not None:
                return scoped

        active_fast_mode = (
            bool(failure_input.active_fast_mode)
            if failure_input.active_fast_mode is not None
            else bool(callback_identity.fast_mode)
        )
        classification = self._completion_policy.resolve_simulation_failure(
            envelope.payload,
            fast_mode=active_fast_mode,
        )
        if classification.status_only_preview:
            return SimulationFailureDecision(
                disposition="status_only_preview_failure",
                envelope=envelope,
                callback_identity=callback_identity,
                freshness=freshness,
                matched_context=context,
                context_mutation=FailureContextMutationRequest(
                    kind="deactivate_if_active",
                    context=context,
                ),
                runtime_consequence=FailureRuntimeConsequenceRequest(kind="preview_failure"),
                ui_consequence=FailureUiConsequenceRequest(
                    kind="current_preview_failure",
                    status_text=str(classification.preview_status_text or ""),
                ),
                display_consequence=FailureDisplayConsequenceRequest(
                    kind="deauthorize_current_preview_failure",
                    target_set_ids=self._current_preview_failure_display_target_set_ids(
                        policy_context=policy_context,
                        callback_identity=callback_identity,
                    ),
                    request_id=int(callback_identity.request_id),
                    run_id=int(callback_identity.run_id),
                    preview_owner_epoch=callback_identity.preview_owner_epoch,
                    status_text=str(classification.preview_status_text or ""),
                    failure_payload=envelope.payload,
                ),
                accepted_runtime_completion=True,
                consumed_runtime_completion=True,
                log_message=classification.error_text,
            )
        return self._terminal_decision(
            failure_input,
            payload=envelope.payload,
            callback_identity=callback_identity,
            context=context,
            freshness=freshness,
            fast_mode=active_fast_mode,
        )

    def _current_preview_failure_display_target_set_ids(
        self,
        *,
        policy_context: CompletionPolicyContext | None,
        callback_identity: SimulationCallbackIdentity,
    ) -> tuple[str, ...]:
        for raw_values in (
            getattr(policy_context, "preview_scope_set_ids", None),
            getattr(policy_context, "explicit_cache_preview_scope_set_ids", None),
            getattr(policy_context, "queue_ids", None),
        ):
            values = normalize_preview_target_set_ids(raw_values or ())
            if values:
                return values

        try:
            ownership = self._completion_policy_preview_ownership()
        except Exception:
            ownership = None
        owner_request_id = getattr(ownership, "request_id", None)
        owner_epoch = getattr(ownership, "epoch", None)
        callback_epoch = getattr(callback_identity, "preview_owner_epoch", None)
        if owner_request_id is not None and int(owner_request_id) == int(callback_identity.request_id):
            epoch_matches = callback_epoch is None or owner_epoch is None or int(owner_epoch) == int(callback_epoch)
            if bool(epoch_matches):
                values = normalize_preview_target_set_ids(getattr(ownership, "target_set_ids", ()) or ())
                if values:
                    return values

        return normalize_preview_target_set_ids((getattr(callback_identity, "batch_set_id", "") or "",))

    def _resolve_stale_runtime_lane_outcome(
        self,
        failure_input: SimulationFailureInput,
    ) -> SimulationFailureDecision:
        envelope = SimulationFailureEnvelope.from_payload(
            failure_input.error_payload,
            origin=failure_input.origin or "runtime_lane",
            callback_identity=failure_input.callback_identity,
            set_id=failure_input.set_id,
            set_name=failure_input.set_name,
        )
        runtime_session_stale = failure_input.runtime_session_stale
        if isinstance(runtime_session_stale, Mapping):
            return SimulationFailureDecision(
                disposition="ignore_stale_runtime_session_outcome",
                envelope=envelope,
                callback_identity=failure_input.callback_identity,
                matched_context=failure_input.context if isinstance(failure_input.context, Mapping) else None,
                accepted_runtime_completion=False,
                consumed_runtime_completion=False,
                log_message=envelope.user_message,
            )
        return SimulationFailureDecision(
            disposition="reset_stale_runtime_lane_outcome",
            envelope=envelope,
            callback_identity=failure_input.callback_identity,
            matched_context=failure_input.context if isinstance(failure_input.context, Mapping) else None,
            accepted_runtime_completion=False,
            consumed_runtime_completion=False,
            stop_current_poll_batch=True,
            log_message=envelope.user_message,
        )

    def _superseded_fast_decision(
        self,
        failure_input: SimulationFailureInput,
        *,
        envelope: SimulationFailureEnvelope,
        callback_identity: SimulationCallbackIdentity,
        context: Mapping[str, Any] | None,
        policy_context: CompletionPolicyContext | None,
        freshness: Any,
    ) -> SimulationFailureDecision:
        request_id = int(callback_identity.request_id)
        superseded = self._completion_policy.resolve_superseded_fast_error(
            preview_ownership=self._completion_policy_preview_ownership(),
            context=policy_context,
            request_id=request_id,
            preview_owner_epoch=getattr(freshness, "callback_preview_owner_epoch", None),
        )
        callback_context_matches_current = bool(
            isinstance(context, Mapping)
            and self._batch_context_owner.context_matches_current_run_identity(context)
        )
        deactivate_current = bool(
            superseded.deactivate_context_immediately and callback_context_matches_current
        )
        reset_status = bool(superseded.reset_status_progress and callback_context_matches_current)
        return SimulationFailureDecision(
            disposition="superseded_fast_failure",
            envelope=envelope,
            callback_identity=callback_identity,
            freshness=freshness,
            matched_context=context if isinstance(context, Mapping) else None,
            context_mutation=FailureContextMutationRequest(
                kind="apply_state_patch",
                context=context if isinstance(context, Mapping) else None,
                base_context=context if isinstance(context, Mapping) else None,
                state_patch=superseded.state_patch,
            ),
            runtime_consequence=FailureRuntimeConsequenceRequest(
                kind="soft_shutdown" if deactivate_current else "none",
                stale_fast_error_replay_needed=True,
            ),
            ui_consequence=FailureUiConsequenceRequest(
                kind="superseded_fast_error",
                status_text="Ready" if reset_status else "",
            ),
            accepted_runtime_completion=False,
            consumed_runtime_completion=True,
            log_message=envelope.user_message,
        )

    def _scoped_failure_decision(
        self,
        failure_input: SimulationFailureInput,
        *,
        envelope: SimulationFailureEnvelope,
        callback_identity: SimulationCallbackIdentity,
        context: Mapping[str, Any],
        freshness: Any,
    ) -> SimulationFailureDecision | None:
        completion_state = self._batch_context_owner.completion_state(context)
        if (
            completion_state is None
            or not completion_state.active
            or not (completion_state.runtime_task_queue or completion_state.parallel)
            or bool(completion_state.fast_mode)
        ):
            return None
        total = max(1, int(completion_state.total or len(completion_state.queue_ids) or 1))
        if total <= 1:
            return None
        sid = str(failure_input.set_id or callback_identity.batch_set_id or envelope.set_id or "")
        completed_ids = {str(item) for item in completion_state.completed_set_ids or () if str(item)}
        completed_ids.add(sid)
        completed_count = len(completed_ids)
        label = str(failure_input.set_name or callback_identity.batch_set or sid or "set")
        final = completed_count >= total
        return SimulationFailureDecision(
            disposition="scoped_failure_final" if final else "scoped_failure_progress",
            envelope=envelope,
            callback_identity=callback_identity,
            freshness=freshness,
            matched_context=context,
            context_mutation=FailureContextMutationRequest(
                kind="record_scoped_failure_and_deactivate" if final else "record_scoped_failure",
                context=context,
                set_id=sid,
                set_name=label,
                failure_payload=envelope.payload,
            ),
            runtime_consequence=FailureRuntimeConsequenceRequest(
                kind="scoped_failure_complete" if final else "none",
                replay_fast_mode=False,
                terminal_replay_needed=bool(final),
            ),
            ui_consequence=FailureUiConsequenceRequest(
                kind="scoped_failure_final" if final else "scoped_failure_progress",
                progress_label=label,
                completed=completed_count,
                total=total,
            ),
            accepted_runtime_completion=True,
            consumed_runtime_completion=True,
            log_message=envelope.user_message,
        )

    def _terminal_decision(
        self,
        failure_input: SimulationFailureInput,
        *,
        payload: object,
        callback_identity: SimulationCallbackIdentity | None,
        context: Mapping[str, Any] | None,
        freshness: Any,
        fast_mode: bool,
        missing_context_reason: str = "",
    ) -> SimulationFailureDecision:
        envelope = SimulationFailureEnvelope.from_payload(
            payload,
            origin=failure_input.origin,
            callback_identity=callback_identity,
            set_id=failure_input.set_id or getattr(callback_identity, "batch_set_id", "") or "",
            set_name=failure_input.set_name or getattr(callback_identity, "batch_set", "") or getattr(callback_identity, "batch_set_id", "") or "",
        )
        cancelled = bool(envelope.cancelled)
        runtime_kind = "stop" if cancelled else "terminal_failure"
        return SimulationFailureDecision(
            disposition="terminal_failure",
            envelope=envelope,
            callback_identity=callback_identity,
            freshness=freshness,
            matched_context=context if isinstance(context, Mapping) else None,
            context_mutation=FailureContextMutationRequest(
                kind="deactivate_if_active" if isinstance(context, Mapping) else "none",
                context=context if isinstance(context, Mapping) else None,
            ),
            runtime_consequence=FailureRuntimeConsequenceRequest(
                kind=runtime_kind,
                replay_fast_mode=bool(fast_mode),
                terminal_replay_needed=not cancelled,
            ),
            ui_consequence=FailureUiConsequenceRequest(kind="terminal_failure"),
            accepted_runtime_completion=False,
            consumed_runtime_completion=False,
            terminal_runtime_completion=True,
            failed_runtime_completion=True,
            log_message=envelope.user_message,
        )
