from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import numpy as np

from kindred.core.batch_cache_contracts import build_overlay_entry
from kindred.gui.ports import (
    ActiveDisplayKind,
    ActiveDisplayTransaction,
    DisplaySetMetadata,
    DisplaySetRole,
    DisplayStatus,
)

__all__ = [
    "active_transaction_for_display_commit",
    "deduped_set_ids",
    "display_overlay_entry",
    "display_species_for_metadata",
    "metadata_for_display_commit",
    "owned_species_for_display_entry",
    "series_for_display_species",
    "transaction_overlay_is_reference",
]


def deduped_set_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values or ():
        value = str(raw_value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def display_species_for_metadata(
    *,
    series: Mapping[str, Any],
    display_species: Sequence[str],
    fallback_names: Sequence[str] = (),
) -> tuple[str, ...]:
    raw_series = {str(name): values for name, values in dict(series or {}).items() if str(name)}
    if not raw_series:
        return ()
    allowed = deduped_set_ids(tuple(str(name) for name in (display_species or ()) if str(name)))
    if not allowed:
        return ()
    requested = deduped_set_ids(tuple(str(name) for name in (fallback_names or ()) if str(name))) or allowed
    allowed_set = set(allowed)
    return tuple(name for name in requested if name in raw_series and name in allowed_set)


def _display_series_for_metadata(
    *,
    series: Mapping[str, Any],
    display_species: Sequence[str],
) -> Dict[str, Any]:
    raw_series = {str(name): values for name, values in dict(series or {}).items() if str(name)}
    display_names = tuple(str(name) for name in (display_species or ()) if str(name))
    return {name: raw_series[name] for name in display_names if name in raw_series}


def series_for_display_species(
    *,
    series: Mapping[str, Any] | None,
    display_species: Sequence[str] | None,
) -> Dict[str, np.ndarray] | None:
    if not isinstance(series, Mapping):
        return None
    names = deduped_set_ids(tuple(str(name) for name in (display_species or ()) if str(name)))
    if not names:
        return None
    raw_series = dict(series or {})
    display_series: Dict[str, np.ndarray] = {}
    for name in names:
        if name not in raw_series:
            return None
        try:
            display_series[name] = np.asarray(raw_series[name], dtype=float).reshape(-1).copy()
        except Exception:
            return None
    return display_series or None


def transaction_overlay_is_reference(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("layer_kind") or "").strip() == "reference"


def owned_species_for_display_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
    raw_owned = entry.get("owned_species")
    return deduped_set_ids(tuple(str(name) for name in (raw_owned or ()) if str(name)))


def display_overlay_entry(
    *,
    label: str,
    entry: Mapping[str, Any],
    set_id: str,
    layer_kind: str,
    layer_id: str,
    owned_species: Sequence[str] | None = None,
    visible: bool | None = None,
) -> Dict[str, object]:
    overlay = dict(
        build_overlay_entry(
            label=label,
            entry=entry,
            set_id=set_id,
            layer_kind=layer_kind,
            layer_id=layer_id,
        )
    )
    raw_display_species = entry.get("display_species")
    if isinstance(raw_display_species, Sequence) and not isinstance(raw_display_species, (str, bytes)):
        overlay["display_species"] = deduped_set_ids(tuple(str(name) for name in raw_display_species if str(name)))
    owned = deduped_set_ids(
        tuple(str(name) for name in (owned_species or ()) if str(name))
    ) or owned_species_for_display_entry(entry)
    display_species = display_species_for_metadata(
        series=overlay.get("series") if isinstance(overlay.get("series"), Mapping) else {},
        display_species=(
            overlay.get("display_species")
            if isinstance(overlay.get("display_species"), Sequence)
            and not isinstance(overlay.get("display_species"), (str, bytes))
            else None
        ),
    )
    if display_species:
        overlay["display_species"] = display_species
    overlay["series"] = _display_series_for_metadata(
        series=overlay.get("series") if isinstance(overlay.get("series"), Mapping) else {},
        display_species=display_species,
    )
    if owned:
        overlay["owned_species"] = owned
    if visible is not None:
        overlay["visible"] = bool(visible)
    completion_provenance = entry.get("completion_provenance")
    if isinstance(completion_provenance, Mapping):
        overlay["completion_provenance"] = dict(completion_provenance)
    return overlay


def metadata_for_display_commit(
    *,
    t: np.ndarray,
    series: Mapping[str, Any],
    overlays: Sequence[Mapping[str, Any]],
    primary_set_id: str,
    primary_label: str,
    owned_species: Sequence[str] | None,
    display_species: Sequence[str],
    completion_provenance: Mapping[str, Any] | None,
    workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, DisplaySetMetadata]:
    primary_owned = deduped_set_ids(tuple(str(name) for name in (owned_species or ()) if str(name)))
    primary_display_species = display_species_for_metadata(series=series, display_species=display_species)
    primary_series = _display_series_for_metadata(
        series=series,
        display_species=primary_display_species,
    )
    primary_id = str(primary_set_id or "").strip()
    primary_layer_id = f"result:{primary_id}" if primary_id else "result:live"
    metadata_by_layer_id: Dict[str, DisplaySetMetadata] = {
        primary_layer_id: DisplaySetMetadata(
            set_id=primary_id,
            label=str(primary_label or primary_id or "Results"),
            role=DisplaySetRole.PRIMARY_RESULT,
            t=t,
            series=primary_series,
            owned_species=primary_owned,
            display_species=primary_display_species,
            layer_id=primary_layer_id,
            completion_provenance=(
                dict(completion_provenance)
                if isinstance(completion_provenance, Mapping)
                else None
            ),
            workspace_preview_provenance=(
                dict((workspace_preview_provenance_by_set_id or {}).get(primary_id) or {})
                if primary_id and isinstance((workspace_preview_provenance_by_set_id or {}).get(primary_id), Mapping)
                else None
            ),
        )
    }
    for raw_overlay in overlays or ():
        if not isinstance(raw_overlay, Mapping):
            continue
        overlay = dict(raw_overlay)
        set_id = str(overlay.get("set_id") or "").strip()
        if not set_id:
            continue
        overlay_series = dict(overlay.get("series") or {})
        overlay_owned = deduped_set_ids(
            tuple(str(name) for name in (overlay.get("owned_species") or ()) if str(name))
        )
        overlay_display_species = display_species_for_metadata(
            series=overlay_series,
            display_species=(
                overlay.get("display_species")
                if isinstance(overlay.get("display_species"), Sequence)
                and not isinstance(overlay.get("display_species"), (str, bytes))
                else None
            ),
        )
        overlay_series = _display_series_for_metadata(
            series=overlay_series,
            display_species=overlay_display_species,
        )
        role = (
            DisplaySetRole.REFERENCE_OVERLAY
            if transaction_overlay_is_reference(overlay)
            else DisplaySetRole.RESULT_OVERLAY
        )
        layer_id = str(overlay.get("layer_id") or f"result:{set_id}")
        metadata_by_layer_id[layer_id] = DisplaySetMetadata(
            set_id=set_id,
            label=str(overlay.get("popup_label") or overlay.get("label") or set_id),
            role=role,
            t=overlay.get("t"),
            series=overlay_series,
            owned_species=overlay_owned,
            display_species=overlay_display_species,
            layer_id=layer_id,
            completion_provenance=(
                dict(overlay.get("completion_provenance"))
                if isinstance(overlay.get("completion_provenance"), Mapping)
                else None
            ),
            workspace_preview_provenance=(
                dict((workspace_preview_provenance_by_set_id or {}).get(set_id) or {})
                if isinstance((workspace_preview_provenance_by_set_id or {}).get(set_id), Mapping)
                else None
            ),
            visible=bool(overlay.get("visible", True)),
        )
    return metadata_by_layer_id


def active_transaction_for_display_commit(
    *,
    t: np.ndarray,
    series: Mapping[str, Any],
    overlays: Sequence[Mapping[str, Any]],
    primary_set_id: str,
    primary_label: str,
    display_set_ids: Sequence[str],
    owned_species: Sequence[str] | None,
    display_species: Sequence[str],
    completion_provenance: Mapping[str, Any] | None,
    workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
    active_kind: ActiveDisplayKind,
    status: DisplayStatus,
    request_id: int | None = None,
    run_id: int | None = None,
    intervention_annotations: Sequence[Mapping[str, Any]] = (),
    show_intervention_annotations: bool = False,
) -> ActiveDisplayTransaction:
    display_ids = deduped_set_ids(display_set_ids)
    transaction_id = ":".join(
        part
        for part in (
            active_kind.value,
            str(request_id if request_id is not None else ""),
            str(run_id if run_id is not None else ""),
            str(primary_set_id or "").strip(),
            ",".join(display_ids),
        )
        if part
    )
    return ActiveDisplayTransaction(
        transaction_id=transaction_id,
        kind=active_kind,
        display_set_ids=display_ids,
        primary_display_set_id=str(primary_set_id or "").strip(),
        sets=metadata_for_display_commit(
            t=t,
            series=series,
            overlays=overlays,
            primary_set_id=primary_set_id,
            primary_label=primary_label,
            owned_species=owned_species,
            display_species=display_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
        ),
        status=status,
        intervention_annotations=tuple(
            dict(item)
            for item in (intervention_annotations or ())
            if isinstance(item, Mapping)
        ),
        show_intervention_annotations=bool(show_intervention_annotations),
    )
