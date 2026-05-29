from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict

import numpy as np

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
    "display_metadata_for_entry",
    "display_species_for_metadata",
    "metadata_for_display_commit",
    "owned_species_for_display_entry",
    "series_for_display_species",
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


def owned_species_for_display_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
    raw_owned = entry.get("owned_species")
    return deduped_set_ids(tuple(str(name) for name in (raw_owned or ()) if str(name)))


def _display_species_from_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
    raw_display_species = entry.get("display_species")
    if not isinstance(raw_display_species, Sequence) or isinstance(raw_display_species, (str, bytes)):
        return ()
    return deduped_set_ids(tuple(str(name) for name in raw_display_species if str(name)))


def display_metadata_for_entry(
    *,
    label: str,
    entry: Mapping[str, Any],
    set_id: str,
    role: DisplaySetRole,
    layer_id: str,
    owned_species: Sequence[str] | None = None,
    visible: bool = True,
    workspace_preview_provenance: Mapping[str, Any] | None = None,
) -> DisplaySetMetadata | None:
    if not isinstance(entry, Mapping):
        return None
    entry_t = entry.get("t")
    if entry_t is None:
        return None
    display_species = _display_species_from_entry(entry)
    display_series = series_for_display_species(
        series=entry.get("series"),
        display_species=display_species,
    )
    if display_series is None:
        return None
    resolved_owned = deduped_set_ids(tuple(str(name) for name in (owned_species or ()) if str(name)))
    if not resolved_owned:
        resolved_owned = owned_species_for_display_entry(entry)
    completion_provenance = entry.get("completion_provenance")
    return DisplaySetMetadata(
        set_id=str(set_id or "").strip(),
        label=str(label or set_id or "Results"),
        role=role,
        t=entry_t,
        series=display_series,
        owned_species=resolved_owned,
        display_species=display_species_for_metadata(
            series=display_series,
            display_species=display_species,
        ),
        layer_id=str(layer_id or "").strip(),
        completion_provenance=(
            dict(completion_provenance)
            if isinstance(completion_provenance, Mapping)
            else None
        ),
        workspace_preview_provenance=(
            dict(workspace_preview_provenance)
            if isinstance(workspace_preview_provenance, Mapping)
            else None
        ),
        visible=bool(visible),
    )


def metadata_for_display_commit(
    *,
    t: np.ndarray,
    series: Mapping[str, Any],
    primary_set_id: str,
    primary_label: str,
    owned_species: Sequence[str] | None,
    display_species: Sequence[str],
    completion_provenance: Mapping[str, Any] | None,
    workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
    additional_metadata: Sequence[DisplaySetMetadata] = (),
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
    for metadata in additional_metadata or ():
        if not isinstance(metadata, DisplaySetMetadata):
            continue
        layer_id = str(metadata.layer_id or "").strip()
        if not layer_id or layer_id == primary_layer_id:
            continue
        metadata_by_layer_id[layer_id] = metadata
    return metadata_by_layer_id


def active_transaction_for_display_commit(
    *,
    t: np.ndarray,
    series: Mapping[str, Any],
    primary_set_id: str,
    primary_label: str,
    display_set_ids: Sequence[str],
    owned_species: Sequence[str] | None,
    display_species: Sequence[str],
    completion_provenance: Mapping[str, Any] | None,
    workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
    active_kind: ActiveDisplayKind,
    status: DisplayStatus,
    additional_metadata: Sequence[DisplaySetMetadata] = (),
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
            primary_set_id=primary_set_id,
            primary_label=primary_label,
            owned_species=owned_species,
            display_species=display_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
            additional_metadata=additional_metadata,
        ),
        status=status,
        intervention_annotations=tuple(
            dict(item)
            for item in (intervention_annotations or ())
            if isinstance(item, Mapping)
        ),
        show_intervention_annotations=bool(show_intervention_annotations),
    )
