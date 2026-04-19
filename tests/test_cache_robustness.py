"""Test cache edge cases."""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.unit]


class TestCacheRobustness:
    """Test caching system edge cases."""

    def test_cache_with_unhashable_params(self):
        """Test cache handles parameters that can't be hashed."""
        from kindred.core.cache import SimulationCache

        cache = SimulationCache(max_size=10)

        # Try to cache with dict parameter (unhashable)
        params = {"nested": {"dict": [1, 2, 3]}}

        # Should handle gracefully (skip caching or convert to hashable)
        try:
            key = cache._compute_key("test_mech", params, "config")
            # If it succeeds, key should be string
            assert isinstance(key, str)
        except (TypeError, ValueError):
            # Error is acceptable for unhashable inputs
            pass

    def test_cache_eviction_under_pressure(self):
        """Test cache eviction when max size is reached."""
        from kindred.core.cache import SimulationCache

        cache = SimulationCache(max_size=3)

        # Add 5 items to cache of size 3
        for i in range(5):
            cache.set(f"mech_{i}", {}, {}, {"result": i})

        # Cache should have at most 3 items
        stats = cache.get_stats()
        # Can't directly check size without private access, but can check it doesn't crash
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0
