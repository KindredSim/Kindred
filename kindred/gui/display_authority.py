from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kindred.gui.ports import (
    DisplayAuthorityBundle,
    DisplayAuthorityInvalidationContext,
    FreshPreviewDisplayEntry,
    ResolvedBatchDisplayRequestEntry,
)


def compose_resolved_display_request_entry(
    *,
    set_id: str,
    label: str,
    active_display_payload: Mapping[str, Any],
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> ResolvedBatchDisplayRequestEntry:
    return ResolvedBatchDisplayRequestEntry(
        set_id=str(set_id),
        label=str(label),
        authority=compose_display_authority_bundle(
            active_display_payload=active_display_payload,
            invalidation_context=invalidation_context,
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
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> FreshPreviewDisplayEntry:
    return FreshPreviewDisplayEntry(
        set_id=str(set_id),
        label=str(label),
        authority=compose_display_authority_bundle(
            active_display_payload=active_display_payload,
            invalidation_context=invalidation_context,
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
    invalidation_context: DisplayAuthorityInvalidationContext | None = None,
) -> DisplayAuthorityBundle:
    return DisplayAuthorityBundle(
        active_display_payload=_copy_display_payload(active_display_payload),
        invalidation_context=(
            invalidation_context
            if isinstance(invalidation_context, DisplayAuthorityInvalidationContext)
            else DisplayAuthorityInvalidationContext()
        ),
    )


def _copy_display_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload or {})
    for key in ("series", "algebra_scalars", "solver_provenance", "completion_provenance"):
        if isinstance(copied.get(key), Mapping):
            copied[key] = dict(copied[key])
    if copied.get("owned_species") is not None:
        copied["owned_species"] = tuple(str(name) for name in copied.get("owned_species") or () if str(name))
    if copied.get("display_species") is not None:
        copied["display_species"] = tuple(str(name) for name in copied.get("display_species") or () if str(name))
    return copied
