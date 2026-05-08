from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.ports import SimulationCacheOpResult


@dataclass(frozen=True, slots=True)
class SimulationCachePublicationResult:
    cache_key: str | None
    cache_token: str | None
    composite_key: str | None


@dataclass
class SimulationCacheAdmin:
    """Owns cache-cap persistence, purge/stats operations, and completion publication."""

    cache: BatchSimulationCache
    settings_set_value: Callable[[str, object], None]
    settings_sync: Callable[[], None]
    record_nonfatal_exception: Callable[[str, BaseException], None]

    def publish_completion_cache(
        self,
        *,
        cache_key: str | None,
        cache_token: str | None,
        set_id: str | None,
        is_preview: bool,
        t: object,
        series: Mapping[str, object],
        algebra_scalars: Mapping[str, object] | None = None,
        mechanism: Any = None,
        mechanism_text: str = "",
        simulation_identity: Mapping[str, Any] | None = None,
        solver_config: Mapping[str, Any] | None = None,
        preview_batch_cache_token: str | None = None,
        fallback_occurred: bool = False,
        fallback_message: Any = None,
        solver_provenance: Mapping[str, Any] | None = None,
        preview_scope_set_ids: Sequence[str] | None = None,
    ) -> SimulationCachePublicationResult:
        normalized_key = str(cache_key or "").strip() or None
        normalized_token = str(cache_token or "").strip() or normalized_key
        normalized_set_id = str(set_id or "").strip() or None
        if normalized_key and bool(is_preview):
            self.cache.record_preview_completion_cache_key(
                cache_key=normalized_key,
                preview_scope_set_ids=preview_scope_set_ids,
            )
        composite_key = None
        if normalized_token and normalized_set_id:
            composite_key = self.cache.put_completion_entry(
                cache_key=normalized_token,
                set_id=normalized_set_id,
                is_preview=bool(is_preview),
                t=t,
                series=series,
                algebra_scalars=algebra_scalars,
                mechanism=mechanism,
                mechanism_text=str(mechanism_text),
                simulation_identity=simulation_identity,
                solver_config=solver_config,
                preview_batch_cache_token=preview_batch_cache_token,
                fallback_occurred=bool(fallback_occurred),
                fallback_message=fallback_message,
                solver_provenance=solver_provenance,
            )
        return SimulationCachePublicationResult(
            cache_key=normalized_key,
            cache_token=normalized_token,
            composite_key=composite_key,
        )

    def publish_completion_cache_truth(
        self,
        *,
        is_preview: bool,
        cache_key: str | None,
        preview_scope_set_ids: Sequence[str] | None = None,
        clear_active_selection_state: bool = False,
        active_cache_key: str | None = None,
        active_cache_preview_token: str | None = None,
        active_cache_preview_scope_set_ids: Sequence[str] | None = None,
        active_cache_valid_set_ids: Sequence[str] | None = None,
        active_cache_invalidated_set_ids: Sequence[str] | None = None,
    ) -> None:
        normalized_key = str(cache_key or "").strip()
        if bool(is_preview):
            if normalized_key:
                self.cache.record_preview_completion_cache_key(
                    cache_key=normalized_key,
                    preview_scope_set_ids=preview_scope_set_ids,
                )
            return
        self.cache.apply_explicit_cache_reconciliation(
            clear_active_selection_state=bool(clear_active_selection_state),
            active_cache_key=active_cache_key,
            active_cache_preview_token=active_cache_preview_token,
            active_cache_preview_scope_set_ids=active_cache_preview_scope_set_ids,
            active_cache_valid_set_ids=active_cache_valid_set_ids,
            active_cache_invalidated_set_ids=active_cache_invalidated_set_ids,
        )

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
            stats = self.cache.stats_best_effort()
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
