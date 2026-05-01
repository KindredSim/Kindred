from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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

    def set_caps(self, *, result_cap: int, preview_cap: int) -> None:
        self.result_cache_cap = max(0, int(result_cap))
        self.preview_cache_cap = max(0, int(preview_cap))
        self.result_cache.set_max_entries(int(self.result_cache_cap))
        self.preview_cache.set_max_entries(int(self.preview_cache_cap))

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

    def clear_active_selection_state(self) -> None:
        self.active_cache_key = None
        self.active_cache_preview_token = None
        self.active_cache_preview_scope_set_ids = None
        self.active_cache_valid_set_ids = None
        self.active_cache_invalidated_set_ids = None
        self.clear_active_preview_selection_state()
        self.clear_display_selection_state()

    def reset_runtime_state(self) -> None:
        self.purge_all()
        self.clear_active_selection_state()
