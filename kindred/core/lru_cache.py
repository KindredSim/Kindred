from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
import sys
from typing import Generic, Optional, TypeVar


K = TypeVar("K")
V = TypeVar("V")


def _clamp_max_entries(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except Exception:
        return 0
    return max(0, n)


def _approx_sizeof(obj: object, *, _seen: Optional[set[int]] = None, _depth: int = 0) -> int:
    """
    Best-effort, bounded approximate size estimator (bytes).

    Goals:
    - Avoid expensive deep walks.
    - Be robust to cycles.
    - Treat NumPy arrays as their `nbytes` payload (plus minimal overhead).
    """
    if obj is None:
        return 0

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)

    # Keep recursion bounded and predictable.
    if _depth > 6:
        try:
            return int(sys.getsizeof(obj))
        except Exception:
            return 0

    total = 0
    try:
        total += int(sys.getsizeof(obj))
    except Exception:
        total += 0

    # Fast-path numpy arrays without importing numpy.
    mod = getattr(getattr(obj, "__class__", object), "__module__", "")
    if isinstance(mod, str) and mod.startswith("numpy") and hasattr(obj, "nbytes"):
        nbytes = getattr(obj, "nbytes", 0)
        try:
            total += int(nbytes)
        except (TypeError, ValueError, OverflowError):
            total += 0
        return total

    # Common containers (bounded traversal).
    try:
        if isinstance(obj, dict):
            for idx, (k, v) in enumerate(obj.items()):
                if idx >= 2000:
                    break
                total += _approx_sizeof(k, _seen=_seen, _depth=_depth + 1)
                total += _approx_sizeof(v, _seen=_seen, _depth=_depth + 1)
            return total
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                if idx >= 2000:
                    break
                total += _approx_sizeof(item, _seen=_seen, _depth=_depth + 1)
            return total
        if isinstance(obj, (set, frozenset)):
            for idx, item in enumerate(obj):
                if idx >= 2000:
                    break
                total += _approx_sizeof(item, _seen=_seen, _depth=_depth + 1)
            return total
    except Exception:
        return total

    return total


class LRUCache(MutableMapping[K, V], Generic[K, V]):
    """
    Deterministic, bounded, in-memory LRU cache.

    - Evicts least-recently-used entries when `max_entries` is exceeded.
    - Provides best-effort size reporting for UI status displays.
    """

    def __init__(self, *, max_entries: int, sizeof: Optional[Callable[[V], int]] = None) -> None:
        self._max_entries = _clamp_max_entries(max_entries)
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._sizeof = sizeof

    def max_entries(self) -> int:
        return int(self._max_entries)

    def set_max_entries(self, max_entries: int) -> list[K]:
        self._max_entries = _clamp_max_entries(max_entries)
        return self._evict_if_needed()

    def used_entries(self) -> int:
        return int(len(self._data))

    def approx_bytes(self) -> int:
        total = 0
        if self._sizeof is not None:
            for v in self._data.values():
                try:
                    total += int(self._sizeof(v))
                except Exception:
                    total += _approx_sizeof(v)
            return int(max(0, total))

        for k, v in self._data.items():
            total += _approx_sizeof(k)
            total += _approx_sizeof(v)
        return int(max(0, total))

    def put(self, key: K, value: V) -> list[K]:
        self._data[key] = value
        self._data.move_to_end(key, last=True)
        return self._evict_if_needed()

    def _evict_if_needed(self) -> list[K]:
        evicted: list[K] = []
        cap = int(self._max_entries)
        if cap <= 0:
            if self._data:
                evicted.extend(list(self._data.keys()))
                self._data.clear()
            return evicted

        while len(self._data) > cap:
            try:
                k, _v = self._data.popitem(last=False)
            except KeyError:
                break
            evicted.append(k)
        return evicted

    # --- MutableMapping ---
    def __getitem__(self, key: K) -> V:
        v = self._data[key]
        self._data.move_to_end(key, last=True)
        return v

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- Convenience ---
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:  # type: ignore[override]
        if key not in self._data:
            return default
        return self.__getitem__(key)

    def clear(self) -> None:  # type: ignore[override]
        self._data.clear()
