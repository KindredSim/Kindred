from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kindred.core.lru_cache import LRUCache


@dataclass
class BatchSimulationCache:
    """
    Owns explicit/preview batch simulation caches and related selection state.

    This object is intentionally Qt-free so it can be exercised in unit tests and
    shared by GUI-facing controllers without living in the GUI layer.
    """

    result_cache_cap: int = 100
    preview_cache_cap: int = 3
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
