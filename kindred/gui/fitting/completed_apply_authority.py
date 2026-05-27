from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from kindred.gui.fitting.constants import INITIAL_PREFIX

PROJECT_APPLY_SCOPE_PARAMETERS = "parameters"
PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS = "initial_conditions"
PROJECT_APPLY_SCOPE_BOTH = "both"


@dataclass(frozen=True)
class CompletedFitApplyAuthority:
    run_stamp_hash: str
    dataset_ids: frozenset[str]
    shared_items: tuple[tuple[str, float], ...]
    dataset_items: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]

    @staticmethod
    def _float_items(params: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
        items: list[tuple[str, float]] = []
        for name, value in params.items():
            try:
                items.append((str(name), float(value)))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(items, key=lambda item: item[0]))

    @classmethod
    def from_result(
        cls,
        result: object,
        *,
        dataset_ids: Sequence[str],
        run_stamp_hash: str,
    ) -> "CompletedFitApplyAuthority":
        normalized_ids = {str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip()}
        dataset_params = getattr(result, "dataset_params", None) or {}
        dataset_items: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        for dataset_id, params in dataset_params.items():
            ds_id = str(dataset_id).strip()
            if not ds_id or not isinstance(params, Mapping):
                continue
            normalized_ids.add(ds_id)
            dataset_items.append((ds_id, cls._float_items(params)))
        return cls(
            run_stamp_hash=str(run_stamp_hash or ""),
            dataset_ids=frozenset(normalized_ids),
            shared_items=cls._float_items(getattr(result, "shared_params", None) or {}),
            dataset_items=tuple(sorted(dataset_items, key=lambda item: item[0])),
        )

    def depends_on_any(self, dataset_ids: Sequence[str]) -> bool:
        remove_set = {str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip()}
        return bool(remove_set and self.dataset_ids.intersection(remove_set))

    def shared_params(self) -> Dict[str, float]:
        return {name: float(value) for name, value in self.shared_items}

    def dataset_params(self) -> Dict[str, Dict[str, float]]:
        return {
            dataset_id: {name: float(value) for name, value in items}
            for dataset_id, items in self.dataset_items
        }

    def initial_condition_params(self) -> Dict[str, Dict[str, float]]:
        initial_params: Dict[str, Dict[str, float]] = {}
        for dataset_id, items in self.dataset_items:
            ds_updates = {
                name: float(value)
                for name, value in items
                if str(name).startswith(INITIAL_PREFIX)
            }
            if ds_updates:
                initial_params[dataset_id] = ds_updates
        return initial_params

    def available_scopes(self) -> set[str]:
        scopes: set[str] = set()
        has_parameters = bool(self.shared_items)
        has_initial_conditions = bool(self.initial_condition_params())
        if has_parameters:
            scopes.add(PROJECT_APPLY_SCOPE_PARAMETERS)
        if has_initial_conditions:
            scopes.add(PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS)
        if has_parameters and has_initial_conditions:
            scopes.add(PROJECT_APPLY_SCOPE_BOTH)
        return scopes


class CompletedFitApplyAuthorityOwner:
    def __init__(self) -> None:
        self._authority: Optional[CompletedFitApplyAuthority] = None

    def set_from_result(
        self,
        result: object,
        *,
        dataset_ids: Sequence[str],
        run_stamp_hash: str,
    ) -> CompletedFitApplyAuthority:
        authority = CompletedFitApplyAuthority.from_result(
            result,
            dataset_ids=dataset_ids,
            run_stamp_hash=run_stamp_hash,
        )
        self._authority = authority
        return authority

    def clear(self) -> Optional[CompletedFitApplyAuthority]:
        authority = self._authority
        self._authority = None
        return authority

    def peek(self) -> Optional[CompletedFitApplyAuthority]:
        return self._authority

    def clear_if_depends_on_any(
        self,
        dataset_ids: Sequence[str],
    ) -> Optional[CompletedFitApplyAuthority]:
        authority = self._authority
        if authority is not None and authority.depends_on_any(dataset_ids):
            self._authority = None
            return authority
        return None

    def current_for_run_stamp(
        self,
        run_stamp_hash: str,
    ) -> Optional[CompletedFitApplyAuthority]:
        authority = self._authority
        if authority is None:
            return None
        active_hash = str(run_stamp_hash or "")
        authority_hash = str(authority.run_stamp_hash or "")
        if not active_hash or not authority_hash or authority_hash != active_hash:
            self._authority = None
            return None
        return authority
