from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.ports import SimulationCacheOpResult


@dataclass
class SimulationCacheAdmin:
    """Owns cache-cap persistence and purge/stats operations for simulation caching."""

    cache: BatchSimulationCache
    settings_set_value: Callable[[str, object], None]
    settings_sync: Callable[[], None]
    record_nonfatal_exception: Callable[[str, BaseException], None]

    def _failure(
        self,
        *,
        operation: str,
        message: str,
        context: str,
        exc: BaseException,
        cache_state_changed: bool = False,
    ) -> SimulationCacheOpResult:
        self.record_nonfatal_exception(context, exc)
        return SimulationCacheOpResult(
            ok=False,
            operation=str(operation),
            message=str(message),
            cache_state_changed=bool(cache_state_changed),
        )

    def set_caps(
        self,
        *,
        result_cap: Any,
        preview_cap: Any,
        persist: bool = True,
    ) -> SimulationCacheOpResult:
        try:
            result_cap_n = max(0, int(result_cap))
        except Exception:
            result_cap_n = int(self.cache.result_cache_cap)
        try:
            preview_cap_n = max(0, int(preview_cap))
        except Exception:
            preview_cap_n = int(self.cache.preview_cache_cap)

        try:
            self.cache.set_caps(result_cap=int(result_cap_n), preview_cap=int(preview_cap_n))
        except Exception as exc:
            return self._failure(
                operation="set_caps",
                message=f"Failed to apply cache caps: {exc}",
                context=(
                    f"Failed to set simulation cache caps "
                    f"(result={int(result_cap_n)} preview={int(preview_cap_n)})"
                ),
                exc=exc,
            )
        if not bool(persist):
            return SimulationCacheOpResult(
                ok=True,
                operation="set_caps",
                cache_state_changed=True,
            )

        try:
            self.settings_set_value("simulation/result_cache_cap", int(result_cap_n))
            self.settings_set_value("simulation/preview_cache_cap", int(preview_cap_n))
            self.settings_sync()
        except Exception as exc:
            return self._failure(
                operation="persist_cache_caps",
                message=f"Applied cache caps, but failed to persist them: {exc}",
                context=(
                    f"Failed to persist simulation cache caps "
                    f"(result={int(result_cap_n)} preview={int(preview_cap_n)})"
                ),
                exc=exc,
                cache_state_changed=True,
            )
        return SimulationCacheOpResult(
            ok=True,
            operation="set_caps",
            cache_state_changed=True,
        )

    def stats(self) -> SimulationCacheOpResult:
        try:
            stats = {
                "result": {
                    "used": int(self.cache.result_cache.used_entries()),
                    "cap": int(self.cache.result_cache.max_entries()),
                    "bytes": max(0, int(self.cache.result_cache.approx_bytes())),
                },
                "preview": {
                    "used": int(self.cache.preview_cache.used_entries()),
                    "cap": int(self.cache.preview_cache.max_entries()),
                    "bytes": max(0, int(self.cache.preview_cache.approx_bytes())),
                },
            }
        except Exception as exc:
            return self._failure(
                operation="stats",
                message=f"Failed to read simulation cache status: {exc}",
                context="Failed to read simulation cache status",
                exc=exc,
            )
        return SimulationCacheOpResult(
            ok=True,
            operation="stats",
            stats=stats,
        )

    def purge_result_cache(self) -> SimulationCacheOpResult:
        try:
            self.cache.purge_result_cache()
        except Exception as exc:
            return self._failure(
                operation="purge_result_cache",
                message=f"Failed to clear simulation result cache: {exc}",
                context="Failed to clear simulation result cache",
                exc=exc,
            )
        return SimulationCacheOpResult(
            ok=True,
            operation="purge_result_cache",
            cache_state_changed=True,
        )

    def purge_preview_cache(self) -> SimulationCacheOpResult:
        try:
            self.cache.purge_preview_cache()
        except Exception as exc:
            return self._failure(
                operation="purge_preview_cache",
                message=f"Failed to clear simulation preview cache: {exc}",
                context="Failed to clear simulation preview cache",
                exc=exc,
            )
        return SimulationCacheOpResult(
            ok=True,
            operation="purge_preview_cache",
            cache_state_changed=True,
        )

    def purge_all_caches(self) -> SimulationCacheOpResult:
        result_outcome = self.purge_result_cache()
        preview_outcome = self.purge_preview_cache()
        if result_outcome.ok and preview_outcome.ok:
            return SimulationCacheOpResult(
                ok=True,
                operation="purge_all_caches",
                cache_state_changed=True,
            )
        messages = [
            outcome.message
            for outcome in (result_outcome, preview_outcome)
            if not outcome.ok and str(outcome.message).strip()
        ]
        return SimulationCacheOpResult(
            ok=False,
            operation="purge_all_caches",
            message="; ".join(messages),
            cache_state_changed=bool(
                result_outcome.cache_state_changed or preview_outcome.cache_state_changed
            ),
        )
