from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .backend import get_symbolic_backend_metadata


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructureCacheKey:
    structure_fingerprint: str
    backend_name: str
    backend_version: str
    profile_version: str


class SymbolicJacobianStructureCache:
    """Caches reusable symbolic Jacobian structures; concrete values bind outside."""

    def __init__(self, *, max_entries: int = 128) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[SymbolicJacobianStructureCacheKey, Any] = OrderedDict()
        self._lock = RLock()

    def get_or_build(
        self,
        key: SymbolicJacobianStructureCacheKey,
        builder: Callable[[], Any],
    ) -> Any:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            built = builder()
            if built is None:
                return None
            self._entries[key] = built
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return built


def symbolic_jacobian_structure_cache_key(
    *,
    structure_fingerprint: str,
) -> SymbolicJacobianStructureCacheKey:
    metadata = get_symbolic_backend_metadata()
    return SymbolicJacobianStructureCacheKey(
        structure_fingerprint=str(structure_fingerprint),
        backend_name=str(metadata.backend_name),
        backend_version=str(metadata.backend_version),
        profile_version=str(metadata.profile_version),
    )


_SYMBOLIC_JACOBIAN_STRUCTURE_CACHE = SymbolicJacobianStructureCache()
def get_or_build_symbolic_jacobian_structure(
    key: SymbolicJacobianStructureCacheKey,
    builder: Callable[[], Any],
) -> Any:
    return _SYMBOLIC_JACOBIAN_STRUCTURE_CACHE.get_or_build(key, builder)
