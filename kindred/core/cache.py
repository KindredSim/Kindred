"""
Simulation result caching for Kindred.

This module provides deterministic caching of simulation results to avoid
redundant computations when the same mechanism and parameters are used.

Scope and intent:
- Optimised for repeated, structurally identical workloads (e.g., re-running
  the same `SimulationRequest`, slider runs that reuse a precompiled RHS, or
  fitting loops that evaluate the same mechanism many times).
- Uses a structural mechanism hash plus a caller-supplied parameter signature
  so cache keys stay stable across in-place parameter tweaks (RateBinding) while
  still separating runs with different numeric inputs.
- Not a general-purpose “make everything faster” switch; results are cached only
  when callers opt in (or provide fingerprints) and side-effectful callbacks
  should bypass caching entirely.

Features:
- Deterministic mechanism hashing (based on species, reactions, equilibria)
- LRU cache with configurable size
- Optional disk-based persistence
- Cache statistics (hits/misses)
- Automatic invalidation on mechanism changes
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import marshal
import os
import pickle  # nosec B403 - used for local disk cache; guarded by file ownership/permission checks (see _is_safe_disk_cache_file)
import stat
import sys
import types
import weakref
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar, ParamSpec, TYPE_CHECKING

import numpy as np

from kindred.core.intervention_schedule import intervention_schedule_identity_fingerprints
from kindred.core.lru_cache import LRUCache
from kindred.core.equilibrium_rate_authority import normalize_existing_equilibrium_rate_authority
from kindred.core.symbolic.artifacts import symbolic_jacobian_identity_payload
from kindred.core.symbolic.identity import normalize_symbolic_identity_mapping, symbolic_identity_json

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from kindred.core.simulator.solvers import SimulationRequest

logger = logging.getLogger(__name__)

__all__ = [
    "cache_simulation",
    "generate_mechanism_hash",
    "fingerprint_simulation_request",
    "clear_cache",
    "get_cache_stats",
    "CacheStats",
    "SimulationCache",
]

# Type hints for decorator
P = ParamSpec('P')
R = TypeVar('R')

# Global cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
}
# Architecture note (lifecycle / leak prevention):
# `@cache_simulation` creates wrapper callables dynamically. We keep a registry of
# wrappers so `clear_cache()` can clear *all* active caches, but this registry
# must not keep wrappers alive forever. Using a WeakSet means that once a wrapper
# loses all strong references, it is eligible for garbage collection and will
# disappear from the registry, preventing unbounded process-lifetime memory
# growth across repeated module reloads / dynamic wrapper creation.
_registered_caches: "weakref.WeakSet[Callable[..., Any]]" = weakref.WeakSet()


class CacheStats:
    """
    Cache performance statistics.

    Attributes
    ----------
    hits : int
        Number of cache hits
    misses : int
        Number of cache misses
    evictions : int
        Number of cache evictions (LRU)
    hit_rate : float
        Cache hit rate (hits / total_requests)
    """

    def __init__(self, hits: int, misses: int, evictions: int):
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, hit_rate={self.hit_rate:.2%})"
        )


class SimulationCache:
    """
    Lightweight in-memory cache for simulation results.

    Provides a minimal API for robustness tests and GUI hooks:
    - Accepts potentially unhashable parameter structures by normalizing them
      to JSON-friendly payloads.
    - Uses an LRU eviction policy bounded by ``max_size``.
    """

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._store: "LRUCache[str, Any]" = LRUCache(max_entries=int(max_size))
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = RLock()

    def _normalize(self, obj: Any) -> Any:
        """Convert unhashable containers into deterministic, JSON-safe forms."""
        if isinstance(obj, dict):
            return {str(k): self._normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, set):
            return sorted((self._normalize(v) for v in obj), key=lambda v: repr(v))
        if isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        return obj

    def _compute_key(self, mechanism_id: Any, params: Any, config: Any) -> str:
        """
        Build a stable hash key from mechanism identifier and parameters.

        Returns a SHA256 hex digest string even if the inputs are normally
        unhashable (dicts, lists, etc.).
        """
        payload = {
            "mechanism": mechanism_id,
            "params": self._normalize(params),
            "config": self._normalize(config),
        }
        try:
            serialized = json.dumps(payload, sort_keys=True, default=str)
        except TypeError:
            serialized = repr(payload)
        return hashlib.sha256(serialized.encode("utf-8", "ignore")).hexdigest()

    def get(self, mechanism_id: Any, params: Any, config: Any) -> Optional[Any]:
        """Retrieve a cached result, returning None on a miss."""
        key = self._compute_key(mechanism_id, params, config)
        with self._lock:
            try:
                value = self._store[key]
            except KeyError:
                self._stats["misses"] += 1
                return None
            else:
                self._stats["hits"] += 1
                return value

    def set(self, mechanism_id: Any, params: Any, config: Any, result: Any) -> str:
        """Store a simulation result and perform LRU eviction if needed."""
        key = self._compute_key(mechanism_id, params, config)
        with self._lock:
            evicted = self._store.put(key, result)
            if evicted:
                self._stats["evictions"] += int(len(evicted))
        return key

    def clear(self) -> None:
        """Remove all cached entries and reset local statistics."""
        with self._lock:
            self._store.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_stats(self) -> Dict[str, int]:
        """Return cache stats dict (hits, misses, evictions)."""
        with self._lock:
            return dict(self._stats)


def get_cache_stats() -> CacheStats:
    """
    Get current cache statistics.

    Returns
    -------
    CacheStats
        Cache performance metrics
    """
    return CacheStats(
        hits=_cache_stats["hits"],
        misses=_cache_stats["misses"],
        evictions=_cache_stats["evictions"],
    )


def clear_cache() -> None:
    """
    Clear all cached simulation results and reset statistics.

    This is useful when:
    - Memory usage is too high
    - Mechanism definitions have changed
    - Testing/debugging cache behavior
    """
    global _cache_stats
    _cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    for cache_wrapper in list(_registered_caches):
        try:
            clear_func = getattr(cache_wrapper, "cache_clear", None)
            if callable(clear_func):
                clear_func()
            else:
                logger.debug("Registered cache wrapper missing cache_clear: %r", cache_wrapper)
        except Exception:
            logger.debug("Failed to clear a registered cache", exc_info=True)
    logger.info("Cache cleared")


def _stable_schedule_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_schedule_structure(value: Any, *, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return float(value)
        return {"kind": "float", "value": repr(value)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            raise TypeError("cyclic mapping state is not cacheable")
        _seen.add(marker)
        try:
            items = [
                [
                    _normalize_schedule_structure(key, _seen=_seen),
                    _normalize_schedule_structure(val, _seen=_seen),
                ]
                for key, val in sorted(value.items(), key=lambda item: repr(item[0]))
            ]
        finally:
            _seen.remove(marker)
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_normalize_schedule_structure(item, _seen=_seen) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_normalize_schedule_structure(item, _seen=_seen) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_normalize_schedule_structure(item, _seen=_seen) for item in value]
        items.sort(key=_stable_schedule_json)
        return {"kind": "set", "items": items}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        marker = id(value)
        if marker in _seen:
            raise TypeError("cyclic dataclass state is not cacheable")
        _seen.add(marker)
        try:
            fields_payload = {
                field.name: _normalize_schedule_structure(getattr(value, field.name), _seen=_seen)
                for field in dataclasses.fields(value)
            }
        finally:
            _seen.remove(marker)
        return {
            "kind": "dataclass",
            "class": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": fields_payload,
        }
    callable_fp = _fingerprint_python_schedule_callable(value, _seen=_seen)
    if callable_fp is not None:
        return {"kind": "callable", "fingerprint": callable_fp}
    raise TypeError(f"Unsupported schedule state type: {type(value)!r}")


def _describe_python_function(func: Any, *, _seen: set[int] | None = None) -> Optional[dict[str, Any]]:
    if isinstance(func, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return None
    code = getattr(func, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    try:
        code_bytes = marshal.dumps(code)
        defaults = _normalize_schedule_structure(getattr(func, "__defaults__", None), _seen=_seen)
        kwdefaults = _normalize_schedule_structure(getattr(func, "__kwdefaults__", None), _seen=_seen)
        closure = getattr(func, "__closure__", None) or ()
        closure_values = [
            _normalize_schedule_structure(cell.cell_contents, _seen=_seen)
            for cell in closure
        ]
    except (TypeError, ValueError):
        return None
    return {
        "kind": "python_function",
        "module": str(getattr(func, "__module__", "") or ""),
        "qualname": str(getattr(func, "__qualname__", getattr(func, "__name__", "")) or ""),
        "code": code_bytes.hex(),
        "defaults": defaults,
        "kwdefaults": kwdefaults,
        "closure": closure_values,
    }


def _describe_callable_instance_state(obj: Any, *, _seen: set[int] | None = None) -> Optional[dict[str, Any]]:
    state_fields: dict[str, Any] = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            state_fields = {
                field.name: _normalize_schedule_structure(getattr(obj, field.name), _seen=_seen)
                for field in dataclasses.fields(obj)
            }
        except TypeError:
            return None
        return {
            "kind": "dataclass_instance",
            "class": f"{type(obj).__module__}:{type(obj).__qualname__}",
            "fields": state_fields,
        }

    try:
        raw_dict = vars(obj)
    except TypeError:
        raw_dict = {}
    if raw_dict:
        try:
            state_fields.update(
                {
                    str(key): _normalize_schedule_structure(val, _seen=_seen)
                    for key, val in sorted(raw_dict.items(), key=lambda item: str(item[0]))
                }
            )
        except TypeError:
            return None

    slots = getattr(type(obj), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if slot in {"__dict__", "__weakref__"} or not hasattr(obj, slot):
            continue
        if slot in state_fields:
            continue
        try:
            state_fields[str(slot)] = _normalize_schedule_structure(getattr(obj, slot), _seen=_seen)
        except TypeError:
            return None

    return {
        "kind": "instance_state",
        "class": f"{type(obj).__module__}:{type(obj).__qualname__}",
        "fields": state_fields,
    }


def _fingerprint_python_schedule_callable(
    val: Any,
    *,
    _seen: set[int] | None = None,
) -> Optional[str]:
    if not callable(val):
        return None
    marker = id(val)
    if _seen is None:
        _seen = set()
    if marker in _seen:
        return None
    _seen.add(marker)
    try:
        if isinstance(val, functools.partial):
            func_desc = _fingerprint_python_schedule_callable(val.func, _seen=_seen)
            if func_desc is None:
                return None
            try:
                args = _normalize_schedule_structure(val.args, _seen=_seen)
                keywords = _normalize_schedule_structure(val.keywords or {}, _seen=_seen)
            except TypeError:
                return None
            return _stable_schedule_json(
                {"kind": "partial", "func": func_desc, "args": args, "keywords": keywords}
            )

        if inspect.ismethod(val):
            func_desc = _describe_python_function(val.__func__, _seen=_seen)
            self_desc = _describe_callable_instance_state(val.__self__, _seen=_seen)
            if func_desc is None or self_desc is None:
                return None
            return _stable_schedule_json(
                {"kind": "bound_method", "func": func_desc, "self": self_desc}
            )

        if inspect.isfunction(val):
            func_desc = _describe_python_function(val, _seen=_seen)
            if func_desc is None:
                return None
            return _stable_schedule_json(func_desc)

        if isinstance(val, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return None

        call_func = getattr(type(val), "__call__", None)
        func_desc = _describe_python_function(call_func, _seen=_seen)
        state_desc = _describe_callable_instance_state(val, _seen=_seen)
        if func_desc is None or state_desc is None:
            return None
        return _stable_schedule_json(
            {
                "kind": "callable_object",
                "class": f"{type(val).__module__}:{type(val).__qualname__}",
                "call": func_desc,
                "state": state_desc,
            }
        )
    finally:
        _seen.remove(marker)


def _fingerprint_temperature_schedule_value(val: Any) -> Optional[str]:
    fp = getattr(val, "fingerprint", None)
    if isinstance(fp, str) and fp:
        return fp
    to_dict = getattr(val, "to_dict", None)
    if callable(to_dict):
        try:
            return json.dumps(to_dict(), sort_keys=True, separators=(",", ":"))
        except Exception:
            return None
    if isinstance(val, Mapping):
        try:
            return json.dumps(val, sort_keys=True, separators=(",", ":"))
        except Exception:
            return None
    return _fingerprint_python_schedule_callable(val)


def generate_mechanism_hash(mechanism: Any) -> str:
    """
    Generate deterministic hash for a mechanism.

    The hash is based on:
    - Species names and initial concentrations
    - Reaction stoichiometry and rate constants
    - Equilibrium constants

    This ensures that equivalent mechanisms produce the same hash,
    enabling cache reuse across sessions.

    Parameters
    ----------
    mechanism : Mechanism
        The reaction mechanism to hash

    Returns
    -------
    str
        Hexadecimal hash string (SHA256, 64 characters)

    Notes
    -----
    The hash is deterministic and platform-independent. Changes to:
    - Species names or initial concentrations
    - Reaction stoichiometry or rates
    - Equilibrium constants
    will produce a different hash, invalidating cached results.

    Examples
    --------
    >>> from kindred.core.mechanism import Mechanism
    >>> mech = Mechanism()
    >>> mech.add_species('A', 1.0)
    >>> mech.add_species('B', 0.0)
    >>> hash1 = generate_mechanism_hash(mech)
    >>> hash2 = generate_mechanism_hash(mech)
    >>> assert hash1 == hash2  # Deterministic
    """
    hasher = hashlib.sha256()

    try:
        from kindred.core.rate_binding import RateBinding  # Local import to avoid cycles
    except Exception:  # pragma: no cover - defensive
        RateBinding = None  # type: ignore

    def _hash_rate_obj(rate_obj: Any) -> str:
        # Keep the binding name stable across in-place value updates
        if RateBinding is not None and isinstance(rate_obj, RateBinding):
            return f"RateBinding:{rate_obj.name}"
        # Callables: prefer their qualified name
        if callable(rate_obj) and hasattr(rate_obj, "__name__"):
            return f"callable:{rate_obj.__name__}"
        return repr(rate_obj)

    def _encode(value: Any) -> bytes:
        try:
            return str(value).encode("utf-8", errors="ignore")
        except Exception:
            return repr(value).encode("utf-8", errors="ignore")

    # Hash species (names + initial concentrations) in deterministic order
    for name in sorted(mechanism.species.keys()):
        species = mechanism.species[name]
        hasher.update(name.encode('utf-8'))
        hasher.update(_encode(species.initial_conc))

    # Hash reactions (physical sides, rate law, net stoichiometry, and rate info)
    for rxn in mechanism.reactions:
        for label, mapping in (
            ("reactants", rxn.reactants),
            ("products", rxn.products),
            ("rate_orders", rxn.rate_orders),
            ("net_stoich", rxn.net_stoich),
        ):
            hasher.update(label.encode("utf-8"))
            for name in sorted(mapping.keys()):
                coeff = mapping[name]
                hasher.update(f"{name}:{coeff}".encode('utf-8'))

        # Hash rate (convert to string representation)
        rate_str = _hash_rate_obj(rxn.rate) if rxn.rate is not None else "None"
        hasher.update(rate_str.encode('utf-8'))

        # Hash overrides
        if rxn.overrides:
            for key in sorted(rxn.overrides.keys()):
                val = rxn.overrides[key]
                hasher.update(f"{key}:{val}".encode('utf-8'))

    # Hash equilibria
    for eq in mechanism.equilibria:
        # Forward stoichiometry
        for name in sorted(eq.stoich_forward.keys()):
            coeff = eq.stoich_forward[name]
            hasher.update(f"fwd:{name}:{coeff}".encode('utf-8'))

        # Back stoichiometry
        for name in sorted(eq.stoich_back.keys()):
            coeff = eq.stoich_back[name]
            hasher.update(f"back:{name}:{coeff}".encode('utf-8'))

        # Equilibrium authority and effective execution identity
        authority = normalize_existing_equilibrium_rate_authority(eq)
        for key, value in authority.identity_items():
            hasher.update(f"eq_authority:{key}".encode("utf-8"))
            hasher.update(_hash_rate_obj(value).encode("utf-8", errors="ignore"))
        hasher.update(f"fast:{eq.fast}".encode('utf-8'))

    # Hash select metadata (temperature, units, schedules) deterministically
    metadata_items = sorted(getattr(mechanism, "metadata", {}) .items())
    for key, val in metadata_items:
        if key == "step_index_map":
            continue
        if key in {"temperature_schedule"}:
            val_repr = _fingerprint_temperature_schedule_value(val)
            if val_repr is None:
                raise TypeError("temperature_schedule is not safely fingerprintable for caching")
        else:
            val_repr = val
        hasher.update(f"meta:{key}".encode("utf-8"))
        hasher.update(_encode(val_repr))

    return hasher.hexdigest()


def cache_simulation(
    maxsize: int = 128,
    cache_dir: Optional[Path] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator for caching simulation results.

    Caches simulation outputs based on mechanism hash and simulation parameters.
    Uses LRU (Least Recently Used) eviction policy.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached results (default: 128)
        Set to None for unlimited cache size (not recommended)
    cache_dir : Path, optional
        Directory for disk-based cache persistence
        If None, uses in-memory cache only

    Returns
    -------
    callable
        Decorated function with caching

    Examples
    --------
    >>> from kindred.core.cache import cache_simulation
    >>> from kindred.core.results import integrate_ctc
    >>>
    >>> @cache_simulation(maxsize=256)
    >>> def my_simulation(mechanism, t_span, **kwargs):
    ...     return integrate_ctc(mechanism, t_span, **kwargs)
    >>>
    >>> # First call: cache miss, runs simulation
    >>> result1 = my_simulation(mechanism, (0, 100))
    >>>
    >>> # Second call with same inputs: cache hit, instant return
    >>> result2 = my_simulation(mechanism, (0, 100))

    Notes
    -----
    - Cache key includes mechanism hash, t_span, solver, tolerances, etc.
    - Thread-safe for multi-threaded simulations
    - Disk cache (if enabled) persists across Python sessions
    - Use clear_cache() to reset cache and statistics
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        disk_dir = Path(cache_dir) if cache_dir is not None else None
        try:
            cache_cap = sys.maxsize if maxsize is None else int(maxsize)
        except Exception:
            cache_cap = 0
        cache_store: "LRUCache[Tuple[str, str], R]" = LRUCache(max_entries=int(cache_cap))
        lock = RLock()

        def _hash_payload(payload: Any) -> Optional[str]:
            try:
                return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()
            except Exception as exc:
                logger.debug("Parameters not serializable for caching: %s", exc)
                return None

        def _serialize_params(args_tail: Tuple[Any, ...], kwargs_map: Dict[str, Any]) -> Optional[str]:
            normalized_kwargs = tuple(sorted(kwargs_map.items()))
            return _hash_payload((args_tail, normalized_kwargs))

        def _clear_local_cache() -> None:
            with lock:
                cache_store.clear()

        @wraps(func)
        def wrapper(
            *args: P.args,
            _cache_fingerprint: Optional[str] = None,
            **kwargs: P.kwargs,
        ) -> R:
            global _cache_stats
            custom_params_hash = _cache_fingerprint

            if len(args) == 0:
                logger.debug("No mechanism provided, skipping cache")
                _cache_stats["misses"] += 1
                return func(*args, **kwargs)

            mechanism = args[0]

            try:
                mech_hash = generate_mechanism_hash(mechanism)
            except Exception as exc:
                logger.debug("Failed to hash mechanism: %s", exc, exc_info=True)
                _cache_stats["misses"] += 1
                return func(*args, **kwargs)

            kwargs_for_fingerprint = dict(kwargs)
            params_hash = custom_params_hash or _serialize_params(tuple(args[1:]), kwargs_for_fingerprint)
            if params_hash is None:
                _cache_stats["misses"] += 1
                return func(*args, **kwargs)

            cache_key = (mech_hash, params_hash)

            with lock:
                try:
                    cached = cache_store[cache_key]
                except KeyError:
                    cached = None
                else:
                    _cache_stats["hits"] += 1
                    logger.debug("Cache hit for mechanism %s", mech_hash[:8])
                    return cached

            disk_result: Optional[R] = None
            if disk_dir is not None:
                disk_result = load_from_disk_cache(mech_hash, disk_dir, params_hash)
                if disk_result is not None:
                    with lock:
                        evicted = cache_store.put(cache_key, disk_result)
                        if evicted:
                            _cache_stats["evictions"] += int(len(evicted))
                    _cache_stats["hits"] += 1
                    logger.debug("Disk cache hit for mechanism %s", mech_hash[:8])
                    return disk_result

            _cache_stats["misses"] += 1
            result = func(*args, **kwargs)

            with lock:
                evicted = cache_store.put(cache_key, result)
                if evicted:
                    _cache_stats["evictions"] += int(len(evicted))

                if disk_dir is not None:
                    try:
                        if os.name != "nt":
                            disk_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                        else:
                            disk_dir.mkdir(parents=True, exist_ok=True)
                        cache_file = disk_dir / f"{mech_hash}_{params_hash}.pkl"
                        if not cache_file.exists():
                            with cache_file.open('wb') as f:
                                pickle.dump(result, f)
                            if os.name != "nt":
                                try:
                                    cache_file.chmod(0o600)
                                except OSError:
                                    pass
                            logger.debug("Saved result to disk cache: %s", cache_file)
                    except Exception as exc:
                        logger.warning("Failed to write disk cache: %s", exc)

            return result

        wrapper.cache_clear = _clear_local_cache  # type: ignore[attr-defined]
        _registered_caches.add(wrapper)

        return wrapper

    return decorator


def _is_safe_disk_cache_file(path: Path) -> bool:
    """
    Return True when a cache file is safe to unpickle.

    This is a defense-in-depth check: Kindred's disk cache is a local optimization,
    not a general deserialization API. We only accept regular, non-symlink files
    that are not writable by group/others, and (on POSIX) are owned by the current
    user. If these checks fail, treat the entry as a cache miss.
    """
    try:
        if path.is_symlink() or (not path.is_file()):
            return False
        st = path.stat()
    except OSError:
        return False
    if os.name != "nt":
        try:
            if hasattr(os, "getuid") and int(st.st_uid) != int(os.getuid()):
                return False
        except Exception:
            return False
        if int(st.st_mode) & int(stat.S_IWGRP | stat.S_IWOTH):
            return False
    return True


def load_from_disk_cache(
    mechanism: Any | str,
    cache_dir: Path,
    params_hash: Optional[str] = None,
) -> Optional[Any]:
    """
    Load cached result from disk.

    Parameters
    ----------
    mechanism : Mechanism or str
        Mechanism instance or precomputed mechanism hash
    cache_dir : Path
        Cache directory
    params_hash : str, optional
        Hash of simulation parameters (solver settings, time grid, etc.)

    Returns
    -------
    result : any or None
        Cached result if found, None otherwise
    """
    try:
        if isinstance(mechanism, str) and len(mechanism) == 64:
            mech_hash = mechanism
        else:
            mech_hash = generate_mechanism_hash(mechanism)
        suffix = f"_{params_hash}" if params_hash else ""
        cache_file = cache_dir / f"{mech_hash}{suffix}.pkl"

        if cache_file.exists():
            if not _is_safe_disk_cache_file(cache_file):
                logger.warning("Refusing to load unsafe disk cache entry: %s", cache_file)
                return None
            with cache_file.open("rb") as f:
                result = pickle.load(f)  # nosec B301 - local cache file guarded by ownership/permissions check
            logger.info(f"Loaded result from disk cache: {cache_file}")
            return result

    except Exception as exc:
        logger.warning(f"Failed to load from disk cache: {exc}")

    return None


def fingerprint_simulation_request(req: "SimulationRequest") -> Optional[str]:
    """
    Build a stable fingerprint for a SimulationRequest-like object.

    The fingerprint captures fields that change simulation outcomes but excludes
    opaque callables (rhs) so it can be reused alongside RateBinding updates.

    Parameters
    ----------
    req : SimulationRequest
        Request carrying rhs, t_span, tolerances, grid, and metadata.

    Returns
    -------
    str or None
        Hex digest suitable for cache keys, or None if serialization fails.
    """
    grid_items = tuple(sorted((req.grid or {}).items()))
    t_eval = getattr(req, "t_eval", None)
    t_eval_tuple: Optional[Tuple[float, ...]] = None
    if t_eval is not None:
        try:
            t_eval_tuple = tuple(float(v) for v in np.asarray(t_eval).reshape(-1))
        except Exception:
            t_eval_tuple = None
    jac_cfg = getattr(req, "rosenbrock_jacobian", None)
    jac_tag = None
    if jac_cfg is not None:
        jac_tag = (
            getattr(jac_cfg, "mode", None),
            getattr(jac_cfg, "ml", None),
            getattr(jac_cfg, "mu", None),
        )
    y0_values = getattr(req, "y0", None)
    if y0_values is None:
        y0_tuple: Tuple[float, ...] = tuple()
    else:
        try:
            y0_tuple = tuple(float(v) for v in y0_values)
        except Exception:
            y0_tuple = tuple()

    schedule_val = getattr(req, "temperature_schedule", None)
    schedule_fp: Optional[str] = None
    if schedule_val is not None:
        schedule_fp = _fingerprint_temperature_schedule_value(schedule_val)
    intervention_schedule_declarative_fp: Optional[str] = None
    intervention_schedule_executable_fp: Optional[str] = None
    intervention_species_names: Tuple[str, ...] = tuple()
    intervention_schedule_val = getattr(req, "intervention_schedule", None)
    if intervention_schedule_val is not None:
        try:
            (
                intervention_schedule_declarative_fp,
                intervention_schedule_executable_fp,
            ) = intervention_schedule_identity_fingerprints(intervention_schedule_val)
        except Exception:
            return None
        if intervention_schedule_declarative_fp or intervention_schedule_executable_fp:
            intervention_species_names = tuple(str(name) for name in (getattr(req, "species_names", None) or ()))

    symbolic_jacobian_identity = symbolic_jacobian_identity_payload(getattr(req, "jacobian_func", None))
    symbolic_wegscheider_identity = normalize_symbolic_identity_mapping(
        getattr(req, "symbolic_wegscheider_identity", None),
        label="symbolic Wegscheider identity",
    )

    payload = (
        tuple(map(float, req.t_span)),
        y0_tuple,
        str(req.solver),
        float(req.rtol),
        float(req.atol),
        grid_items,
        t_eval_tuple,
        (
            symbolic_identity_json(symbolic_jacobian_identity).decode("ascii")
            if symbolic_jacobian_identity
            else bool(getattr(req, "jacobian_func", None))
        ),
        (
            symbolic_identity_json(symbolic_wegscheider_identity).decode("ascii")
            if symbolic_wegscheider_identity
            else None
        ),
        jac_tag,
        schedule_fp,
        intervention_schedule_declarative_fp,
        intervention_schedule_executable_fp,
        intervention_species_names,
        bool(getattr(req, "progress_callback", None)),
        getattr(req, "positivity", None),
        tuple(getattr(req, "pos_indices", []) or []),
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


# Example usage (in docstring)
_example_usage = '''
# Example: Cache simulation results

from kindred.core.cache import cache_simulation, get_cache_stats, clear_cache
from kindred.core.results import integrate_ctc

# Decorate simulation function
@cache_simulation(maxsize=256)
def run_simulation(mechanism, t_span=(0, 100), solver='BDF', rtol=1e-6):
    return integrate_ctc(mechanism, t_span, solver=solver, rtol=rtol)

# First run: cache miss
result1 = run_simulation(my_mechanism)

# Second run with same params: cache hit (instant)
result2 = run_simulation(my_mechanism)

# Check cache performance
stats = get_cache_stats()
print(stats)  # CacheStats(hits=1, misses=1, evictions=0, hit_rate=50.00%)

# Clear cache
clear_cache()
'''
