from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kindred.core.batch_cache_contracts import (
    BatchCacheEntryReadResult,
    BatchCacheEntryV1,
    build_batch_cache_entry,
    read_batch_cache_entry,
)
from kindred.core.lru_cache import LRUCache
from kindred.core.runtime_defaults import PREVIEW_CACHE_CAP_DEFAULT, RESULT_CACHE_CAP_DEFAULT


@dataclass
class BatchSimulationCache:
    """
    Owns explicit/preview batch simulation caches and related selection state.

    This object is intentionally Qt-free so it can be exercised in unit tests and
    shared by GUI-facing controllers without living in the GUI layer.
    """

    result_cache_cap: int = RESULT_CACHE_CAP_DEFAULT
    preview_cache_cap: int = PREVIEW_CACHE_CAP_DEFAULT
    result_cache: LRUCache[str, Dict[str, Any]] = field(init=False)
    preview_cache: LRUCache[str, Dict[str, Any]] = field(init=False)

    active_cache_key: Optional[str] = None
    active_cache_preview_token: Optional[str] = None
    active_cache_preview_scope_set_ids: Optional[tuple[str, ...]] = None
    active_cache_valid_set_ids: Optional[tuple[str, ...]] = None
    active_cache_invalidated_set_ids: Optional[tuple[str, ...]] = None
    active_preview_cache_key: Optional[str] = None
    active_preview_scope_set_ids: Optional[tuple[str, ...]] = None
    last_display_selection: List[str] = field(default_factory=list)

    active_batch_set: Optional[str] = None
    active_batch_set_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.result_cache_cap = max(0, int(self.result_cache_cap))
        self.preview_cache_cap = max(0, int(self.preview_cache_cap))
        self.result_cache = LRUCache(max_entries=int(self.result_cache_cap))
        self.preview_cache = LRUCache(max_entries=int(self.preview_cache_cap))

    @staticmethod
    def entry_key(cache_key: str, set_id: str) -> str:
        return f"{str(cache_key)}::{str(set_id)}"

    @staticmethod
    def normalize_set_ids(set_ids: Sequence[str] | None) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            ordered.append(set_id)
        return tuple(ordered)

    def active_result_cache_set_ids(self) -> tuple[str, ...]:
        active_key = str(self.active_cache_key or "").strip()
        if not active_key:
            return ()
        prefix = f"{active_key}::"
        cached_ids: list[str] = []
        try:
            for raw_key in self.result_cache:
                key_s = str(raw_key or "")
                if not key_s.startswith(prefix):
                    continue
                set_id = str(key_s[len(prefix):] or "").strip()
                if set_id and set_id not in cached_ids:
                    cached_ids.append(set_id)
        except Exception:
            cached_ids = []
        if cached_ids:
            return tuple(cached_ids)
        return self.normalize_set_ids(self.active_cache_valid_set_ids)

    def record_active_result_cache_staleness(
        self,
        *,
        set_ids: Sequence[str] = (),
        is_global: bool = False,
    ) -> tuple[str, ...]:
        active_scope = self.active_result_cache_set_ids()
        if bool(is_global):
            stale_scope = active_scope
        else:
            stale_scope = self.normalize_set_ids(
                (*self.normalize_set_ids(self.active_cache_invalidated_set_ids), *self.normalize_set_ids(set_ids))
            )
            if active_scope:
                active_ids = set(active_scope)
                stale_scope = tuple(set_id for set_id in stale_scope if set_id in active_ids)
        self.active_cache_invalidated_set_ids = stale_scope or None
        return stale_scope

    def apply_active_cache_preview_reconciliation(
        self,
        *,
        valid_set_ids: Sequence[str],
        invalidated_set_ids: Sequence[str],
        preview_scope_set_ids: Sequence[str],
        preview_token: str | None,
    ) -> None:
        self.active_cache_valid_set_ids = self.normalize_set_ids(valid_set_ids) or None
        self.active_cache_invalidated_set_ids = self.normalize_set_ids(invalidated_set_ids) or None
        self.active_cache_preview_scope_set_ids = self.normalize_set_ids(preview_scope_set_ids) or None
        self.active_cache_preview_token = str(preview_token or "").strip() or None

    def apply_run_start_cache_decision(
        self,
        *,
        fast_mode: bool,
        explicit_cache_valid_set_ids: Sequence[str] | None = None,
        explicit_cache_invalidated_set_ids: Sequence[str] | None = None,
        preview_scope_set_ids: Sequence[str] | None = None,
    ) -> None:
        if bool(fast_mode):
            self.active_preview_scope_set_ids = self.normalize_set_ids(preview_scope_set_ids) or None
            return
        self.active_cache_preview_scope_set_ids = None
        self.active_cache_valid_set_ids = self.normalize_set_ids(explicit_cache_valid_set_ids) or None
        self.active_cache_invalidated_set_ids = self.normalize_set_ids(explicit_cache_invalidated_set_ids) or None

    def record_preview_completion_cache_key(
        self,
        *,
        cache_key: str,
        preview_scope_set_ids: Sequence[str] | None = None,
    ) -> Optional[str]:
        normalized_key = str(cache_key or "").strip()
        if not normalized_key:
            return None
        self.active_preview_cache_key = normalized_key
        self.active_preview_scope_set_ids = self.normalize_set_ids(preview_scope_set_ids) or None
        return normalized_key

    def record_run_cache_key(
        self,
        *,
        cache_key: str,
        fast_mode: bool,
    ) -> Optional[str]:
        normalized_key = str(cache_key or "").strip()
        if not normalized_key:
            return None
        if bool(fast_mode):
            self.active_preview_cache_key = normalized_key
        else:
            self.active_cache_key = normalized_key
            self.active_cache_preview_token = None
        return normalized_key

    def record_explicit_scoped_failure_cache_state(
        self,
        *,
        cache_key: str,
        explicit_cache_valid_set_ids: Sequence[str] | None,
        explicit_cache_invalidated_set_ids: Sequence[str] | None,
    ) -> bool:
        normalized_key = str(cache_key or "").strip()
        if not normalized_key or str(self.active_cache_key or "") != normalized_key:
            return False
        self.active_cache_valid_set_ids = self.normalize_set_ids(explicit_cache_valid_set_ids) or None
        self.active_cache_invalidated_set_ids = (
            self.normalize_set_ids(explicit_cache_invalidated_set_ids) or None
        )
        return True

    def apply_explicit_cache_reconciliation(
        self,
        *,
        clear_active_selection_state: bool,
        active_cache_key: str | None,
        active_cache_preview_token: str | None,
        active_cache_preview_scope_set_ids: Sequence[str] | None,
        active_cache_valid_set_ids: Sequence[str] | None,
        active_cache_invalidated_set_ids: Sequence[str] | None,
    ) -> None:
        if bool(clear_active_selection_state):
            self.clear_active_selection_state()
            return
        self.active_cache_key = str(active_cache_key or "").strip() or None
        self.active_cache_preview_token = str(active_cache_preview_token or "").strip() or None
        self.active_cache_preview_scope_set_ids = (
            self.normalize_set_ids(active_cache_preview_scope_set_ids) or None
        )
        self.active_cache_valid_set_ids = self.normalize_set_ids(active_cache_valid_set_ids) or None
        self.active_cache_invalidated_set_ids = self.normalize_set_ids(active_cache_invalidated_set_ids) or None

    def put_batch_cache_entry(
        self,
        *,
        cache_key: str,
        set_id: str,
        entry: Mapping[str, Any],
        is_preview: bool,
    ) -> Optional[str]:
        normalized_key = str(cache_key or "").strip()
        normalized_set_id = str(set_id or "").strip()
        if not normalized_key or not normalized_set_id:
            return None
        composite_key = self.entry_key(normalized_key, normalized_set_id)
        cache_store = self.preview_cache if bool(is_preview) else self.result_cache
        cache_store.put(composite_key, dict(entry))
        return composite_key

    def store_for_kind(self, kind: str):
        normalized = str(kind or "").strip().lower()
        if normalized in {"preview", "fast"}:
            return self.preview_cache
        return self.result_cache

    def store_for_preview(self, *, is_preview: bool):
        return self.preview_cache if bool(is_preview) else self.result_cache

    def entry_for_set(
        self,
        *,
        cache_key: str,
        set_id: str,
        is_preview: bool,
    ) -> BatchCacheEntryReadResult:
        normalized_key = str(cache_key or "").strip()
        normalized_set_id = str(set_id or "").strip()
        if not normalized_key or not normalized_set_id:
            return BatchCacheEntryReadResult("missing")
        store = self.store_for_preview(is_preview=bool(is_preview))
        payload = (store or {}).get(self.entry_key(normalized_key, normalized_set_id))
        return read_batch_cache_entry(payload)

    def entries_for_cache_key(
        self,
        *,
        cache_key: str,
        is_preview: bool,
    ) -> tuple[tuple[str, BatchCacheEntryV1], ...]:
        normalized_key = str(cache_key or "").strip()
        if not normalized_key:
            return ()
        store = self.store_for_preview(is_preview=bool(is_preview))
        prefix = f"{normalized_key}::"
        entries: list[tuple[str, BatchCacheEntryV1]] = []
        for raw_key in list((store or {}).keys()):
            key = str(raw_key or "")
            if not key.startswith(prefix):
                continue
            set_id = str(key[len(prefix):] or "").strip()
            result = read_batch_cache_entry((store or {}).get(raw_key))
            if set_id and result.state == "valid" and result.entry is not None:
                entries.append((set_id, result.entry))
        return tuple(sorted(entries, key=lambda item: item[0]))

    @staticmethod
    def _entry_key_suffixes(*, set_id: str = "", set_name: str = "") -> tuple[str, ...]:
        suffixes: list[str] = []
        for raw in (set_id, set_name):
            value = str(raw or "").strip()
            if value:
                suffix = f"::{value}"
                if suffix not in suffixes:
                    suffixes.append(suffix)
        return tuple(suffixes)

    def contains_set_identifier(self, *, set_id: str = "", set_name: str = "") -> bool:
        suffixes = self._entry_key_suffixes(set_id=set_id, set_name=set_name)
        if not suffixes:
            return False
        for store in (self.result_cache, self.preview_cache):
            for raw_key in list((store or {}).keys()):
                key = str(raw_key or "")
                if any(key.endswith(suffix) for suffix in suffixes):
                    return True
        return False

    def purge_entries_for_set_identifiers(
        self,
        *,
        set_ids: Sequence[str] | None = None,
        set_names: Sequence[str] | None = None,
    ) -> int:
        targets = {
            str(value or "").strip()
            for value in (*(set_ids or ()), *(set_names or ()))
            if str(value or "").strip()
        }
        if not targets:
            return 0
        removed = 0
        for store in (self.result_cache, self.preview_cache):
            for raw_key in list((store or {}).keys()):
                key = str(raw_key or "")
                if "::" not in key:
                    continue
                _prefix, suffix = key.rsplit("::", 1)
                if suffix not in targets:
                    continue
                try:
                    del store[raw_key]
                    removed += 1
                except KeyError:
                    continue
        return int(removed)

    def put_completion_entry(
        self,
        *,
        cache_key: str,
        set_id: str,
        is_preview: bool,
        t: object,
        series: Mapping[str, object],
        algebra_scalars: Optional[Mapping[str, object]] = None,
        mechanism: Any = None,
        mechanism_text: str = "",
        simulation_identity: Optional[Mapping[str, Any]] = None,
        solver_config: Optional[Mapping[str, Any]] = None,
        preview_batch_cache_token: Optional[str] = None,
        fallback_occurred: bool = False,
        fallback_message: Any = None,
        solver_provenance: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Sequence[Mapping[str, Any]]] = None,
        completion_provenance: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        entry = build_batch_cache_entry(
            t=t,
            series=series,
            algebra_scalars=algebra_scalars,
            mechanism=mechanism,
            mechanism_text=mechanism_text,
            simulation_identity=simulation_identity,
            solver_config=solver_config,
            preview_batch_cache_token=preview_batch_cache_token,
            fallback_occurred=bool(fallback_occurred),
            fallback_message=fallback_message,
            solver_provenance=solver_provenance,
            warnings=warnings,
            completion_provenance=completion_provenance,
        )
        return self.put_batch_cache_entry(
            cache_key=cache_key,
            set_id=set_id,
            entry=entry,
            is_preview=bool(is_preview),
        )

    def set_caps(self, *, result_cap: int, preview_cap: int) -> None:
        self.result_cache_cap = max(0, int(result_cap))
        self.preview_cache_cap = max(0, int(preview_cap))
        self.result_cache.set_max_entries(int(self.result_cache_cap))
        self.preview_cache.set_max_entries(int(self.preview_cache_cap))

    def result_cache_max_entries(self) -> int:
        return int(self.result_cache.max_entries())

    def preview_cache_max_entries(self) -> int:
        return int(self.preview_cache.max_entries())

    def stats_best_effort(self) -> Dict[str, Dict[str, int]]:
        try:
            r_used = int(self.result_cache.used_entries())
            r_cap = int(self.result_cache.max_entries())
            r_bytes = int(self.result_cache.approx_bytes())
        except Exception:
            r_used, r_cap, r_bytes = (0, 0, 0)
        try:
            p_used = int(self.preview_cache.used_entries())
            p_cap = int(self.preview_cache.max_entries())
            p_bytes = int(self.preview_cache.approx_bytes())
        except Exception:
            p_used, p_cap, p_bytes = (0, 0, 0)
        return {
            "result": {"used": r_used, "cap": r_cap, "bytes": max(0, r_bytes)},
            "preview": {"used": p_used, "cap": p_cap, "bytes": max(0, p_bytes)},
        }

    def purge_result_cache(self) -> None:
        self.result_cache.clear()

    def purge_preview_cache(self) -> None:
        self.preview_cache.clear()

    def purge_all(self) -> None:
        self.purge_result_cache()
        self.purge_preview_cache()

    def clear_active_preview_selection_state(self) -> None:
        self.active_preview_cache_key = None
        self.active_preview_scope_set_ids = None

    def clear_display_selection_state(self) -> None:
        self.last_display_selection.clear()
        self.active_batch_set = None
        self.active_batch_set_id = None

    def clear_active_cache_identity_state(self) -> None:
        self.active_cache_key = None
        self.active_cache_preview_token = None
        self.active_cache_preview_scope_set_ids = None
        self.active_cache_valid_set_ids = None
        self.active_cache_invalidated_set_ids = None
        self.clear_active_preview_selection_state()

    def clear_active_selection_state(self) -> None:
        self.clear_active_cache_identity_state()
        self.clear_display_selection_state()

    def reset_runtime_state(self) -> None:
        self.purge_all()
        self.clear_active_selection_state()
