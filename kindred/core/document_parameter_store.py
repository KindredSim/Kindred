"""
Canonical parameter ownership for a Kindred document.

This is intentionally Qt-free so it can be exercised in unit tests and
shared by GUI-facing controllers without living in the GUI layer.

Holds:
- shared_params: document-level parameter values (shared by default)
- set_local_overrides: per-set parameter diffs (optionally set-local)
- effective_params(set_id): composed values
- param_fingerprint(set_id): deterministic hash for cache identity
- schema_id: hash of schema text for cache identity

Concentrations remain per-set and are NOT managed here.
"""

from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional


class DocumentParameterStore:
    """
    Canonical parameter ownership for shared and per-set parameter values.

    This is the single source of truth for parameter state in the document.
    """

    __slots__ = ("_shared_params", "_set_local_overrides", "_schema_text", "_schema_id")

    def __init__(self) -> None:
        self._shared_params: Dict[str, float] = {}
        self._set_local_overrides: Dict[str, Dict[str, float]] = {}
        self._schema_text: str = ""
        self._schema_id: str = ""

    @staticmethod
    def _normalized_local_override_set_id(set_id: Optional[str]) -> Optional[str]:
        """Return a valid per-set override key, or None for the shared/no-set sentinel."""
        set_id_s = str(set_id or "").strip()
        return set_id_s or None

    # ------------------------------------------------------------------
    # Schema identity
    # ------------------------------------------------------------------

    @property
    def schema_text(self) -> str:
        return self._schema_text

    @property
    def schema_id(self) -> str:
        return self._schema_id

    def set_schema(self, text: str) -> None:
        """Update the canonical schema text and recompute the schema identity hash."""
        self._schema_text = str(text or "")
        self._schema_id = hashlib.sha256(self._schema_text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Shared parameters
    # ------------------------------------------------------------------

    @property
    def shared_params(self) -> Mapping[str, float]:
        """Immutable snapshot of the shared parameter baseline."""
        return MappingProxyType(dict(self._shared_params))

    def clear_shared_params(self) -> None:
        self._shared_params.clear()

    # ------------------------------------------------------------------
    # Per-set local overrides
    # ------------------------------------------------------------------

    def _visible_local_override_items(self) -> Iterator[tuple[str, Dict[str, float]]]:
        for raw_set_id, workspace in self._set_local_overrides.items():
            set_id = self._normalized_local_override_set_id(raw_set_id)
            if set_id is None or not workspace:
                continue
            yield set_id, workspace

    @property
    def set_local_overrides(self) -> Dict[str, Dict[str, float]]:
        """Read-only access to the per-set parameter overrides."""
        return {
            set_id: {str(k): float(v) for k, v in dict(workspace).items()}
            for set_id, workspace in self._visible_local_override_items()
        }

    def local_overrides_for_set(self, set_id: Optional[str]) -> Dict[str, float]:
        """Return a copy of the per-set overrides for *set_id*."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return {}
        return dict(self._set_local_overrides.get(set_id_s) or {})

    def set_ids_with_local_overrides(self) -> list[str]:
        return [set_id for set_id, _workspace in self._visible_local_override_items()]

    def has_any_local_overrides(self) -> bool:
        return any(True for _set_id, _workspace in self._visible_local_override_items())

    def has_local_overrides_for_set(self, set_id: Optional[str]) -> bool:
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return False
        return bool(self._set_local_overrides.get(set_id_s))

    def clear_local_overrides_for_set(self, set_id: Optional[str]) -> bool:
        """Remove overrides for *set_id*. Returns True if there was something to remove."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None or set_id_s not in self._set_local_overrides:
            return False
        self._set_local_overrides.pop(set_id_s, None)
        return True

    def clear_all_local_overrides(self) -> None:
        self._set_local_overrides.clear()

    # ------------------------------------------------------------------
    # Effective parameter composition
    # ------------------------------------------------------------------

    def effective_params(self, set_id: Optional[str] = "") -> Dict[str, float]:
        """Return shared params merged with the per-set overrides for *set_id*."""
        result = dict(self._shared_params)
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return result
        overrides = self._set_local_overrides.get(set_id_s)
        if overrides:
            result.update(overrides)
        return result

    # ------------------------------------------------------------------
    # Staging (per-set override mutation)
    # ------------------------------------------------------------------

    def stage_override(self, set_id: Optional[str], name: str, value: float) -> bool:
        """
        Stage a single per-set parameter override.

        If the value matches the shared baseline within tolerance, the override
        is pruned (the parameter returns to the shared value for that set).

        Returns True if the store actually changed.
        """
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return False
        name_s = str(name)
        value_f = float(value)
        baseline = self._shared_params.get(name_s)
        workspace = dict(self._set_local_overrides.get(set_id_s) or {})
        changed = False

        if baseline is not None and math.isclose(value_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
            if name_s in workspace:
                workspace.pop(name_s, None)
                changed = True
        else:
            current = workspace.get(name_s)
            if current is None or not math.isclose(float(current), value_f, rel_tol=1e-12, abs_tol=1e-12):
                workspace[name_s] = value_f
                changed = True

        if workspace:
            self._set_local_overrides[set_id_s] = {
                str(k): float(v) for k, v in workspace.items()
            }
        else:
            self._set_local_overrides.pop(set_id_s, None)

        return changed

    # ------------------------------------------------------------------
    # Shared parameter sync (with pruning)
    # ------------------------------------------------------------------

    def sync_shared_params(self, values: Dict[str, float]) -> bool:
        """
        Replace the shared baseline and prune per-set overrides.

        Per-set override entries that now match the new baseline (within tolerance)
        or reference parameters no longer in the baseline are removed.

        Returns True if per-set overrides changed as a side effect.
        """
        committed: Dict[str, float] = {}
        for key, value in dict(values or {}).items():
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(parsed):
                continue
            committed[str(key)] = float(parsed)

        before = {
            str(sid): {str(k): float(v) for k, v in ws.items()}
            for sid, ws in self._set_local_overrides.items()
        }

        self._shared_params = committed

        pruned: Dict[str, Dict[str, float]] = {}
        for sid, workspace in list(self._set_local_overrides.items()):
            sid_s = self._normalized_local_override_set_id(sid)
            if sid_s is None:
                continue
            filtered: Dict[str, float] = {}
            for name, val in list(dict(workspace or {}).items()):
                baseline = committed.get(str(name))
                if baseline is None:
                    continue
                val_f = float(val)
                if math.isclose(val_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
                    continue
                filtered[str(name)] = val_f
            if filtered:
                pruned[sid_s] = filtered
        self._set_local_overrides = pruned

        after = {
            str(sid): {str(k): float(v) for k, v in ws.items()}
            for sid, ws in self._set_local_overrides.items()
        }
        return before != after

    # ------------------------------------------------------------------
    # Commit / globalize
    # ------------------------------------------------------------------

    def commit_effective_as_shared(self, set_id: Optional[str]) -> Dict[str, float]:
        """
        Merge the focused set's effective values into shared params and preserve
        other sets' staged local overrides where they still differ from the new baseline.

        This is the "commit current set" operation: the focused set's effective values
        become the new shared baseline, the focused set's local overrides are discarded,
        and other sets remain dirty if their staged values still differ from the new
        canonical baseline.
        """
        set_id_s = self._normalized_local_override_set_id(set_id)
        if set_id_s is None:
            return dict(self._shared_params)
        effective = self.effective_params(set_id_s)
        new_shared_params = {
            str(k): float(v) for k, v in dict(effective or {}).items()
        }
        preserved: Dict[str, Dict[str, float]] = {}
        for raw_sid, workspace in list(self._set_local_overrides.items()):
            sid_s = self._normalized_local_override_set_id(raw_sid)
            if sid_s is None or sid_s == set_id_s:
                continue
            filtered: Dict[str, float] = {}
            for name, val in list(dict(workspace or {}).items()):
                baseline = new_shared_params.get(str(name))
                val_f = float(val)
                if baseline is None:
                    filtered[str(name)] = val_f
                    continue
                if math.isclose(val_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
                    continue
                filtered[str(name)] = val_f
            if filtered:
                preserved[sid_s] = filtered
        self._shared_params = new_shared_params
        self._set_local_overrides = preserved
        return dict(self._shared_params)

    # ------------------------------------------------------------------
    # Fingerprinting (for cache identity)
    # ------------------------------------------------------------------

    def param_fingerprint(self, set_id: Optional[str] = "") -> str:
        """Deterministic hash of effective parameter values for cache identity."""
        set_id_s = self._normalized_local_override_set_id(set_id)
        effective = dict(self._shared_params) if set_id_s is None else self.effective_params(set_id_s)
        items = sorted(effective.items())
        material = ";".join(f"{k}={v!r}" for k, v in items)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Bulk reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all parameter state."""
        self._shared_params.clear()
        self._set_local_overrides.clear()
        self._schema_text = ""
        self._schema_id = ""
