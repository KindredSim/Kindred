from __future__ import annotations

import pytest


@pytest.mark.unit
def test_lru_cache_eviction_is_deterministic():
    from kindred.core.lru_cache import LRUCache

    cache: LRUCache[str, int] = LRUCache(max_entries=2)
    cache["a"] = 1
    cache["b"] = 2

    # Touch "a" so "b" becomes the LRU.
    assert cache["a"] == 1

    cache["c"] = 3

    assert "b" not in cache
    assert list(cache.keys()) == ["a", "c"]


@pytest.mark.unit
def test_lru_cache_can_shrink_and_reports_bytes_safely():
    import numpy as np

    from kindred.core.lru_cache import LRUCache

    cache: LRUCache[str, object] = LRUCache(max_entries=3)
    cache["x"] = {"arr": np.zeros((10,), dtype=float)}
    cache["y"] = {"arr": np.zeros((20,), dtype=float)}
    cache["z"] = {"arr": np.zeros((30,), dtype=float)}

    used, cap = cache.used_entries(), cache.max_entries()
    assert used == 3
    assert cap == 3

    approx = cache.approx_bytes()
    assert isinstance(approx, int)
    assert approx >= 0

    evicted = cache.set_max_entries(1)
    assert isinstance(evicted, list)
    assert len(cache) == 1
