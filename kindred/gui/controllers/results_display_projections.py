from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Sequence

import numpy as np

from kindred.gui.controllers.results_display_builders import display_species_for_metadata
from kindred.gui.ports import (
    ActiveDisplayKind,
    ActiveDisplayTransaction,
    CopyAllDisplayBlock,
    CopyAllExportPlan,
    CopyAllMissingItem,
    DisplayEventKind,
    DisplaySetMetadata,
    DisplaySetRole,
    DisplayStatus,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
)

__all__ = [
    "build_copy_all_export_plan",
    "build_main_plot_csv_export",
    "cache_resolution_cause_for_transition",
    "display_mapping_payload",
    "display_status_for_unpublished_request",
    "display_status_is_displayed",
    "display_transaction_provenance_payload",
    "display_transition_status_text",
    "ordered_display_transaction_metadata",
    "published_display_status_text",
]

_COMPLETED_RUN_DENIED_NO_CACHE_STATUS = (
    "Requested set has no cached result; active completed-run display was not replaced."
)
_COMPLETED_RUN_DISPLAY_SCOPE_REMOVED_STATUS = "Displayed completed run removed from display scope."
_COMPLETED_RUN_DELETED_SET_STATUS = "Displayed completed run included a deleted set; display cleared."
_COMPLETED_RUN_RUNTIME_INPUT_CHANGED_STATUS = "Runtime input changed; completed run display cleared."
_COMPLETED_RUN_DISPLAY_UNAVAILABLE_STATUS = "Not displayed."
_SEMANTIC_DISPLAY_UNAVAILABLE_STATUS = "Completed run has no semantic displayable result."


def ordered_display_transaction_metadata(
    transaction: ActiveDisplayTransaction,
) -> list[DisplaySetMetadata]:
    return sorted(
        list(dict(transaction.sets or {}).values()),
        key=lambda item: (
            0 if item.role is DisplaySetRole.PRIMARY_RESULT else 1,
            0 if item.role is DisplaySetRole.RESULT_OVERLAY else 1,
            str(item.layer_id or ""),
        ),
    )


def published_display_status_text(display_status: DisplayStatus) -> str:
    if display_status is DisplayStatus.DISPLAYED_COMPLETED_RUN:
        return "Displayed completed run."
    if display_status is DisplayStatus.DISPLAYED_FRESH_PREVIEW:
        return "Displayed fresh preview."
    if display_status is DisplayStatus.DISPLAYED_DIRECT_RESULT:
        return "Displayed simulation result."
    if display_status is DisplayStatus.DISPLAYED_WORKSPACE_PREVIEW:
        return "Displayed workspace preview."
    if display_status is DisplayStatus.DISPLAYED_RESOLVED_RESULT:
        return "Displayed resolved result."
    if display_status is DisplayStatus.DISPLAYED_CACHED_RESULT:
        return "Displayed cached result."
    return "Displayed result."


def display_status_is_displayed(display_status: DisplayStatus) -> bool:
    return display_status in {
        DisplayStatus.DISPLAYED_COMPLETED_RUN,
        DisplayStatus.DISPLAYED_CACHED_RESULT,
        DisplayStatus.DISPLAYED_RESOLVED_RESULT,
        DisplayStatus.DISPLAYED_WORKSPACE_PREVIEW,
        DisplayStatus.DISPLAYED_FRESH_PREVIEW,
        DisplayStatus.DISPLAYED_DIRECT_RESULT,
    }


def display_status_for_unpublished_request(
    *,
    requested_status: DisplayStatus,
    active_transaction: ActiveDisplayTransaction | None,
) -> DisplayStatus:
    _ = active_transaction
    return requested_status


def display_transition_status_text(
    transition_outcome: DisplayTransitionOutcome | None,
) -> str:
    if (
        isinstance(transition_outcome, DisplayTransitionOutcome)
        and transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
    ):
        base_status = published_display_status_text(transition_outcome.display_status)
        return _with_request_outcome_status_suffix(base_status, transition_outcome)
    cause = transition_outcome.cause if isinstance(transition_outcome, DisplayTransitionOutcome) else None
    event_kind = transition_outcome.event_kind if isinstance(transition_outcome, DisplayTransitionOutcome) else None
    has_active_display = (
        isinstance(transition_outcome, DisplayTransitionOutcome)
        and isinstance(transition_outcome.active_transaction, ActiveDisplayTransaction)
    )
    if (
        isinstance(transition_outcome, DisplayTransitionOutcome)
        and transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
        and transition_outcome.display_status is DisplayStatus.DISPLAY_DENIED
        and has_active_display
        and transition_outcome.cause
        in {
            DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
            DisplayTransitionCause.NO_DISPLAYABLE_PREVIEW_RESULTS,
        }
    ):
        return _with_request_outcome_status_suffix("Display unchanged.", transition_outcome)
    if cause is DisplayTransitionCause.DISPLAY_MUTATION_FAILED:
        return _with_request_outcome_status_suffix("Display failed.", transition_outcome)
    if cause is DisplayTransitionCause.QUEUED_DISPLAY:
        return _with_request_outcome_status_suffix("Display queued.", transition_outcome)
    if cause is DisplayTransitionCause.DISPLAY_MUTATION_DENIED:
        if has_active_display:
            return _with_request_outcome_status_suffix(
                "Display unchanged.",
                transition_outcome,
            )
        return _with_request_outcome_status_suffix(
            "Requested result not displayed.",
            transition_outcome,
        )
    if cause is DisplayTransitionCause.INVALID_CACHE_ENTRY:
        return _with_request_outcome_status_suffix(
            "Cached result invalid.",
            transition_outcome,
        )
    if cause is DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE:
        return _with_request_outcome_status_suffix(
            "Preview pending.",
            transition_outcome,
        )
    if cause is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE:
        return _with_request_outcome_status_suffix(_SEMANTIC_DISPLAY_UNAVAILABLE_STATUS, transition_outcome)
    if cause is DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS:
        return _with_request_outcome_status_suffix(
            _COMPLETED_RUN_DISPLAY_UNAVAILABLE_STATUS,
            transition_outcome,
        )
    if cause is DisplayTransitionCause.NO_DISPLAYABLE_PREVIEW_RESULTS:
        return _with_request_outcome_status_suffix(
            "Preview has no displayable result.",
            transition_outcome,
        )
    if cause is DisplayTransitionCause.SHOW_REMOVED_ACTIVE_SET:
        return _with_request_outcome_status_suffix(
            _COMPLETED_RUN_DISPLAY_SCOPE_REMOVED_STATUS,
            transition_outcome,
        )
    if cause is DisplayTransitionCause.DELETED_ACTIVE_SET:
        return _with_request_outcome_status_suffix(
            _COMPLETED_RUN_DELETED_SET_STATUS,
            transition_outcome,
        )
    if cause is DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY:
        if event_kind is DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE:
            return _with_request_outcome_status_suffix(
                _COMPLETED_RUN_DISPLAY_UNAVAILABLE_STATUS,
                transition_outcome,
            )
        return _with_request_outcome_status_suffix(
            _COMPLETED_RUN_RUNTIME_INPUT_CHANGED_STATUS,
            transition_outcome,
        )
    return _with_request_outcome_status_suffix(
        "Result not cached.",
        transition_outcome,
    )


def _with_request_outcome_status_suffix(
    base_status: str,
    transition_outcome: DisplayTransitionOutcome | None,
) -> str:
    if not isinstance(transition_outcome, DisplayTransitionOutcome):
        return base_status
    suffix = _request_outcome_status_suffix(transition_outcome)
    if not suffix:
        return base_status
    return f"{base_status.rstrip('.')}; {suffix}."


def _request_outcome_status_suffix(
    transition_outcome: DisplayTransitionOutcome,
) -> str:
    failed_ids = tuple(str(set_id) for set_id in (transition_outcome.failed_intent_set_ids or ()) if str(set_id))
    missing_ids = tuple(str(set_id) for set_id in (transition_outcome.missing_intent_set_ids or ()) if str(set_id))
    semantic_ids = tuple(
        str(set_id)
        for set_id in (transition_outcome.semantic_unavailable_set_ids or ())
        if str(set_id)
    )
    categorized_ids = {*failed_ids, *missing_ids, *semantic_ids}
    unresolved_ids = tuple(
        str(set_id)
        for set_id in (transition_outcome.unresolved_intent_set_ids or ())
        if str(set_id) and str(set_id) not in categorized_ids
    )
    fragments: list[str] = []
    labels_by_set_id = {
        str(set_id): str(label)
        for set_id, label in dict(transition_outcome.requested_labels_by_set_id or {}).items()
        if str(set_id) and str(label)
    }
    failed_fragment = _request_outcome_fragment(
        failed_ids,
        labels_by_set_id=labels_by_set_id,
        label_singular_suffix="failed",
        label_plural_suffix="failed",
        count_singular="1 result failed",
        count_plural="{count} results failed",
    )
    if failed_fragment:
        fragments.append(failed_fragment)
    missing_fragment = _request_outcome_fragment(
        missing_ids,
        labels_by_set_id=labels_by_set_id,
        label_singular_suffix="needs a run",
        label_plural_suffix="need a run",
        count_singular="1 result needs a run",
        count_plural="{count} results need a run",
    )
    if missing_fragment:
        fragments.append(missing_fragment)
    semantic_fragment = _request_outcome_fragment(
        semantic_ids,
        labels_by_set_id=labels_by_set_id,
        label_singular_suffix="has no displayable result",
        label_plural_suffix="have no displayable result",
        count_singular="1 result has no displayable result",
        count_plural="{count} results have no displayable result",
    )
    if semantic_fragment:
        fragments.append(semantic_fragment)
    unresolved_fragment = _request_outcome_fragment(
        unresolved_ids,
        labels_by_set_id=labels_by_set_id,
        label_singular_suffix="is unavailable",
        label_plural_suffix="are unavailable",
        count_singular="1 result is unavailable",
        count_plural="{count} results are unavailable",
    )
    if unresolved_fragment:
        fragments.append(unresolved_fragment)
    return _join_status_fragments(fragments)


def _request_outcome_fragment(
    set_ids: Sequence[str],
    *,
    labels_by_set_id: Mapping[str, str],
    label_singular_suffix: str,
    label_plural_suffix: str,
    count_singular: str,
    count_plural: str,
) -> str:
    ids = tuple(str(set_id) for set_id in set_ids if str(set_id))
    if not ids:
        return ""
    labels = tuple(str(labels_by_set_id.get(set_id) or "") for set_id in ids)
    if len(ids) <= 2 and labels and all(labels) and sum(len(label) for label in labels) <= 32:
        label_text = _join_status_fragments(labels)
        suffix = label_singular_suffix if len(labels) == 1 else label_plural_suffix
        return f"{label_text} {suffix}"
    count = len(ids)
    if count == 1:
        return count_singular
    return count_plural.format(count=count)


def _join_status_fragments(fragments: Sequence[str]) -> str:
    items = [str(fragment).strip() for fragment in fragments if str(fragment).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def cache_resolution_cause_for_transition(
    transition_outcome: DisplayTransitionOutcome | None,
    *,
    default: DisplayTransitionCause = DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
) -> DisplayTransitionCause:
    cause = transition_outcome.cause if isinstance(transition_outcome, DisplayTransitionOutcome) else None
    if cause is DisplayTransitionCause.INVALID_CACHE_ENTRY:
        return DisplayTransitionCause.INVALID_CACHE_ENTRY
    if cause is DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE:
        return DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE
    if cause is DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE:
        return DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
    return default


def display_mapping_payload(value: Mapping[str, Any] | None) -> Dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        str(key): (
            display_mapping_payload(raw)
            if isinstance(raw, Mapping)
            else raw
        )
        for key, raw in dict(value).items()
        if str(key)
    }


def display_transaction_provenance_payload(
    transaction: ActiveDisplayTransaction | None,
) -> Dict[str, object]:
    if not isinstance(transaction, ActiveDisplayTransaction):
        return {}
    display_sets: list[Dict[str, object]] = []
    for metadata in ordered_display_transaction_metadata(transaction):
        display_sets.append(
            {
                "set_id": str(metadata.set_id or ""),
                "label": str(metadata.label or metadata.set_id or ""),
                "role": metadata.role.value if isinstance(metadata.role, DisplaySetRole) else str(metadata.role),
                "layer_id": str(metadata.layer_id or ""),
                "visible": bool(metadata.visible),
                "owned_species": list(metadata.owned_species or ()),
                "display_species": list(metadata.display_species or ()),
                "workspace_preview_provenance": display_mapping_payload(
                    metadata.workspace_preview_provenance
                ),
                "num_points": int(
                    np.asarray(
                        metadata.t if metadata.t is not None else [],
                        dtype=float,
                    )
                    .reshape(-1)
                    .size
                ),
            }
        )
    return {
        "display_transaction": {
            "transaction_id": str(transaction.transaction_id or ""),
            "kind": transaction.kind.value if isinstance(transaction.kind, ActiveDisplayKind) else str(transaction.kind),
            "status": transaction.status.value if isinstance(transaction.status, DisplayStatus) else str(transaction.status),
            "display_set_ids": list(transaction.display_set_ids or ()),
            "primary_display_set_id": str(transaction.primary_display_set_id or ""),
        },
        "display_sets": display_sets,
    }


def _copy_all_export_set_id(metadata: DisplaySetMetadata) -> str:
    if metadata.role is DisplaySetRole.REFERENCE_OVERLAY and metadata.set_id:
        return f"{metadata.set_id}:canonical_reference"
    return str(metadata.set_id)


def _copy_all_export_label(metadata: DisplaySetMetadata) -> str:
    return str(metadata.label or metadata.set_id or "Results")


def _csv_export_label(metadata: DisplaySetMetadata) -> str:
    return str(metadata.label or metadata.set_id or metadata.layer_id or "Results")


def _copy_all_missing_reason_from_metadata(metadata: DisplaySetMetadata) -> str:
    t = np.asarray(metadata.t if metadata.t is not None else [], dtype=float).reshape(-1)
    series_raw = metadata.series or {}
    if t.size <= 0 or not isinstance(series_raw, Mapping):
        return "no_simulation_data"
    display_species = display_species_for_metadata(
        series=series_raw,
        display_species=metadata.display_species,
    )
    if not display_species:
        return "no_visible_series"
    return "unavailable"


def _copy_all_display_block_from_metadata(metadata: DisplaySetMetadata) -> object | None:
    t = np.asarray(metadata.t if metadata.t is not None else [], dtype=float).reshape(-1)
    series_raw = metadata.series or {}
    if t.size <= 0 or not isinstance(series_raw, Mapping):
        return None
    display_species = display_species_for_metadata(
        series=series_raw,
        display_species=metadata.display_species,
    )
    series = {}
    for species_name in display_species:
        if species_name not in series_raw:
            continue
        series[species_name] = np.asarray(series_raw[species_name], dtype=float).reshape(-1)
    if not series:
        return None
    return CopyAllDisplayBlock(
        set_id=_copy_all_export_set_id(metadata),
        label=_copy_all_export_label(metadata),
        t=t,
        series=series,
        layer_id=str(metadata.layer_id or ""),
        owned_species=tuple(str(name) for name in (metadata.owned_species or ()) if str(name)),
        display_species=display_species,
    )


def build_copy_all_export_plan(active_transaction: ActiveDisplayTransaction | None) -> object | None:
    if active_transaction is None:
        return None
    display_blocks: list[object] = []
    missing_items: list[object] = []
    for metadata in ordered_display_transaction_metadata(active_transaction):
        if not bool(metadata.visible):
            continue
        block = _copy_all_display_block_from_metadata(metadata)
        if block is not None:
            display_blocks.append(block)
            continue
        label = _copy_all_export_label(metadata)
        missing_items.append(
            CopyAllMissingItem(
                set_id=_copy_all_export_set_id(metadata),
                label=label,
                popup_label=label,
                reason=_copy_all_missing_reason_from_metadata(metadata),
            )
        )
    return CopyAllExportPlan(display_blocks=display_blocks, missing_items=missing_items)


def build_main_plot_csv_export(
    *,
    active_transaction: ActiveDisplayTransaction,
    scope: str,
    axis_state: Mapping[str, object],
) -> tuple[list[str], list[list[object]]]:
    normalized_scope = str(scope or "axis")
    x_name = str(axis_state.get("x_name") or "t")
    x_header = str(axis_state.get("x_header") or x_name)
    requested_y_names = tuple(str(name) for name in (axis_state.get("y_names") or ()) if str(name))
    columns: list[tuple[str, np.ndarray]] = []
    missing_display_sets: list[str] = []
    for metadata in ordered_display_transaction_metadata(active_transaction):
        if not bool(metadata.visible):
            continue
        label = _csv_export_label(metadata)
        t_array = np.asarray(metadata.t if metadata.t is not None else [], dtype=float).reshape(-1)
        if t_array.size <= 0:
            missing_display_sets.append(f"{label} (no time axis)")
            continue
        if not isinstance(metadata.series, Mapping):
            missing_display_sets.append(f"{label} (no series data)")
            continue
        series_map = dict(metadata.series or {})
        if not series_map:
            missing_display_sets.append(f"{label} (no series data)")
            continue
        fallback_names = (
            requested_y_names
            if normalized_scope == "axis"
            else tuple(metadata.display_species)
        )
        species_names = list(
            display_species_for_metadata(
                series=series_map,
                display_species=metadata.display_species,
                fallback_names=fallback_names,
            )
        )
        if fallback_names and not species_names:
            missing_display_sets.append(f"{label} (no selected display series)")
            continue
        if not species_names:
            missing_display_sets.append(f"{label} (no display series)")
            continue
        if x_name == "t":
            x_array = t_array
        else:
            x_array = np.asarray(series_map.get(x_name) if x_name in series_map else [], dtype=float).reshape(-1)
        if x_array.size <= 0:
            missing_display_sets.append(f"{label} (missing X axis '{x_name}')")
            continue
        prefix = "" if metadata.role is DisplaySetRole.PRIMARY_RESULT else f"{label}::"
        metadata_columns: list[tuple[str, np.ndarray]] = [(f"{prefix}{x_header}", x_array)]
        invalid_series: list[str] = []
        valid_series = False
        expected_len = x_array.shape[0]
        for species_name in species_names:
            values = np.asarray(series_map[species_name], dtype=float).reshape(-1)
            if values.shape[0] != expected_len:
                invalid_series.append(species_name)
                continue
            metadata_columns.append((f"{prefix}[{species_name}]", values))
            valid_series = True
        if invalid_series:
            missing_display_sets.append(
                f"{label} (series length mismatch: {', '.join(invalid_series)})"
            )
            continue
        if not valid_series:
            missing_display_sets.append(f"{label} (no exportable display series)")
            continue
        columns.extend(metadata_columns)
    if missing_display_sets:
        detail = "; ".join(missing_display_sets)
        raise ValueError(
            "Cannot export active simulation display CSV because visible display sets are incomplete: "
            + detail
        )
    if not columns:
        raise ValueError("No active simulation display series are available to export.")
    max_len = max(values.shape[0] for _, values in columns)
    header = [name for name, _ in columns]
    rows: list[list[object]] = []
    for idx in range(max_len):
        row: list[object] = []
        for _, values in columns:
            row.append(values[idx] if idx < values.shape[0] else "")
        rows.append(row)
    return header, rows
