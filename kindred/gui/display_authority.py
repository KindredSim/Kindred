from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from kindred.gui.ports import (
    CanonicalReferenceEligibility,
    DisplayAuthorityBundle,
    DisplayAuthorityInvalidationContext,
    FreshPreviewDisplayEntry,
    RequestScopeRestoreTruth,
    ResolvedBatchDisplayRequestEntry,
)


@dataclass(frozen=True, slots=True)
class CanonicalReferenceAuthorityResolution:
    canonical_reference_candidate: Mapping[str, Any] | None
    canonical_reference_eligible_for_current_inputs: bool
    invalidation_context: DisplayAuthorityInvalidationContext


def resolve_canonical_reference_authority(
    *,
    set_id: str,
    active_cache_key: str,
    active_cache_valid_set_ids: object = (),
    active_cache_invalidated_set_ids: object = (),
    workspace_preview_provenance: Mapping[str, Any] | None = None,
    load_canonical_reference_candidate: Callable[[], Mapping[str, Any] | None] | None = None,
) -> CanonicalReferenceAuthorityResolution:
    normalized_set_id = str(set_id or "").strip()
    normalized_cache_key = str(active_cache_key or "").strip()
    valid_set_ids = {
        str(candidate_set_id)
        for candidate_set_id in (active_cache_valid_set_ids or ())
        if str(candidate_set_id)
    }
    invalidated_set_ids = {
        str(candidate_set_id)
        for candidate_set_id in (active_cache_invalidated_set_ids or ())
        if str(candidate_set_id)
    }
    active_cache_set_is_invalidated = bool(normalized_set_id in invalidated_set_ids)
    active_cache_set_is_valid = (
        None
        if not normalized_cache_key
        else (
            (not valid_set_ids and not invalidated_set_ids)
            or normalized_set_id in valid_set_ids
        )
    )
    allow_canonical_candidate_fetch = bool(normalized_cache_key) and (
        active_cache_set_is_valid is True or active_cache_set_is_invalidated
    )
    canonical_reference_candidate = None
    if allow_canonical_candidate_fetch and callable(load_canonical_reference_candidate):
        candidate = load_canonical_reference_candidate()
        if isinstance(candidate, Mapping):
            canonical_reference_candidate = candidate
    return CanonicalReferenceAuthorityResolution(
        canonical_reference_candidate=canonical_reference_candidate,
        canonical_reference_eligible_for_current_inputs=bool(
            canonical_reference_candidate is not None
            and isinstance(workspace_preview_provenance, Mapping)
            and bool(workspace_preview_provenance)
        ),
        invalidation_context=DisplayAuthorityInvalidationContext(
            active_cache_key=normalized_cache_key,
            active_cache_set_is_valid=active_cache_set_is_valid,
            active_cache_set_is_invalidated=active_cache_set_is_invalidated,
        ),
    )


def compose_resolved_display_request_entry(
    *,
    set_id: str,
    label: str,
    active_display_payload: Mapping[str, Any],
    canonical_reference_candidate: Mapping[str, Any] | None = None,
    canonical_reference_eligible_for_current_inputs: bool = False,
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
    request_scope_restore_truth: RequestScopeRestoreTruth = RequestScopeRestoreTruth.NONE,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> ResolvedBatchDisplayRequestEntry:
    return ResolvedBatchDisplayRequestEntry(
        set_id=str(set_id),
        label=str(label),
        authority=compose_display_authority_bundle(
            active_display_payload=active_display_payload,
            canonical_reference_candidate=canonical_reference_candidate,
            canonical_reference_eligible_for_current_inputs=canonical_reference_eligible_for_current_inputs,
            invalidation_context=invalidation_context,
            request_scope_restore_truth=request_scope_restore_truth,
            workspace_preview_provenance=workspace_preview_provenance,
        ),
        workspace_preview_provenance=(
            dict(workspace_preview_provenance)
            if isinstance(workspace_preview_provenance, Mapping)
            else None
        ),
    )


def compose_fresh_preview_display_entry(
    *,
    set_id: str,
    label: str,
    active_display_payload: Mapping[str, Any],
    canonical_reference_candidate: Mapping[str, Any] | None = None,
    canonical_reference_eligible_for_current_inputs: bool = False,
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> FreshPreviewDisplayEntry:
    return FreshPreviewDisplayEntry(
        set_id=str(set_id),
        label=str(label),
        authority=compose_display_authority_bundle(
            active_display_payload=active_display_payload,
            canonical_reference_candidate=canonical_reference_candidate,
            canonical_reference_eligible_for_current_inputs=canonical_reference_eligible_for_current_inputs,
            invalidation_context=invalidation_context,
            request_scope_restore_truth=RequestScopeRestoreTruth.NONE,
            workspace_preview_provenance=workspace_preview_provenance,
        ),
        workspace_preview_provenance=(
            dict(workspace_preview_provenance)
            if isinstance(workspace_preview_provenance, Mapping)
            else None
        ),
    )


def compose_display_authority_bundle(
    *,
    active_display_payload: Mapping[str, Any],
    canonical_reference_candidate: Mapping[str, Any] | None = None,
    canonical_reference_eligible_for_current_inputs: bool = False,
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
    request_scope_restore_truth: RequestScopeRestoreTruth = RequestScopeRestoreTruth.NONE,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> DisplayAuthorityBundle:
    active_payload = _copy_display_payload(active_display_payload)
    candidate_payload = (
        _copy_display_payload(canonical_reference_candidate)
        if isinstance(canonical_reference_candidate, Mapping)
        else None
    )
    canonical_reference_eligibility = CanonicalReferenceEligibility.UNAVAILABLE
    canonical_reference_payload = None
    if candidate_payload is not None:
        if _display_payloads_match(active_payload, candidate_payload):
            canonical_reference_eligibility = CanonicalReferenceEligibility.SAME_AS_ACTIVE_RESULT
            canonical_reference_payload = candidate_payload
        elif bool(canonical_reference_eligible_for_current_inputs):
            canonical_reference_eligibility = CanonicalReferenceEligibility.PROVEN
            canonical_reference_payload = candidate_payload
    return DisplayAuthorityBundle(
        active_display_payload=active_payload,
        canonical_reference_payload=canonical_reference_payload,
        canonical_reference_eligible_for_current_inputs=bool(
            canonical_reference_eligible_for_current_inputs
        ),
        canonical_reference_eligibility=canonical_reference_eligibility,
        invalidation_context=(
            invalidation_context
            if isinstance(invalidation_context, DisplayAuthorityInvalidationContext)
            else DisplayAuthorityInvalidationContext()
        ),
        request_scope_restore_truth=request_scope_restore_truth,
    )


def _copy_display_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload or {})
    if isinstance(copied.get("series"), Mapping):
        copied["series"] = dict(copied["series"])
    if isinstance(copied.get("algebra_scalars"), Mapping):
        copied["algebra_scalars"] = dict(copied["algebra_scalars"])
    if isinstance(copied.get("solver_provenance"), Mapping):
        copied["solver_provenance"] = dict(copied["solver_provenance"])
    if isinstance(copied.get("completion_provenance"), Mapping):
        copied["completion_provenance"] = dict(copied["completion_provenance"])
    if copied.get("owned_species") is not None:
        copied["owned_species"] = tuple(str(name) for name in copied.get("owned_species") or () if str(name))
    if copied.get("display_species") is not None:
        copied["display_species"] = tuple(str(name) for name in copied.get("display_species") or () if str(name))
    return copied


def _display_payloads_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_t = np.asarray(left.get("t") if left.get("t") is not None else [], dtype=float).reshape(-1)
    right_t = np.asarray(right.get("t") if right.get("t") is not None else [], dtype=float).reshape(-1)
    if left_t.shape != right_t.shape or not np.array_equal(left_t, right_t):
        return False
    left_series = left.get("series") if isinstance(left.get("series"), Mapping) else {}
    right_series = right.get("series") if isinstance(right.get("series"), Mapping) else {}
    if set(left_series) != set(right_series):
        return False
    for name in left_series:
        left_values = np.asarray(left_series.get(name) if left_series.get(name) is not None else [], dtype=float)
        right_values = np.asarray(right_series.get(name) if right_series.get(name) is not None else [], dtype=float)
        if left_values.shape != right_values.shape or not np.array_equal(left_values, right_values):
            return False
    return True
