from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from .backend import get_symbolic_backend_metadata


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructureCacheKey:
    structure_fingerprint: str
    solver: str
    wegscheider_cyclicity_enabled: bool
    backend_name: str
    backend_version: str
    profile_version: str


@dataclass(frozen=True, slots=True)
class SymbolicJacobianStructureCacheStats:
    entries: int
    hits: int
    misses: int
    evictions: int


class SymbolicJacobianStructureCache:
    """Caches reusable symbolic Jacobian structures; concrete values bind outside."""

    def __init__(self, *, max_entries: int = 128) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[SymbolicJacobianStructureCacheKey, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_build(
        self,
        key: SymbolicJacobianStructureCacheKey,
        builder: Callable[[], Any],
    ) -> Any:
        if key in self._entries:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self._misses += 1
        built = builder()
        if built is None:
            return None
        self._entries[key] = built
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        return built

    def clear(self) -> None:
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> SymbolicJacobianStructureCacheStats:
        return SymbolicJacobianStructureCacheStats(
            entries=len(self._entries),
            hits=int(self._hits),
            misses=int(self._misses),
            evictions=int(self._evictions),
        )


def symbolic_jacobian_structure_cache_key(
    *,
    structure_fingerprint: str,
    solver: str,
    temperature_K: float,
    wegscheider_cyclicity_enabled: bool,
) -> SymbolicJacobianStructureCacheKey:
    metadata = get_symbolic_backend_metadata()
    return SymbolicJacobianStructureCacheKey(
        structure_fingerprint=str(structure_fingerprint),
        solver=str(solver),
        wegscheider_cyclicity_enabled=bool(wegscheider_cyclicity_enabled),
        backend_name=str(metadata.backend_name),
        backend_version=str(metadata.backend_version),
        profile_version=str(metadata.profile_version),
    )


_SYMBOLIC_JACOBIAN_STRUCTURE_CACHE = SymbolicJacobianStructureCache()


def clear_symbolic_jacobian_structure_cache() -> None:
    _SYMBOLIC_JACOBIAN_STRUCTURE_CACHE.clear()


def symbolic_jacobian_structure_cache_stats() -> SymbolicJacobianStructureCacheStats:
    return _SYMBOLIC_JACOBIAN_STRUCTURE_CACHE.stats()


def get_or_build_symbolic_jacobian_structure(
    key: SymbolicJacobianStructureCacheKey,
    builder: Callable[[], Any],
) -> Any:
    return _SYMBOLIC_JACOBIAN_STRUCTURE_CACHE.get_or_build(key, builder)
